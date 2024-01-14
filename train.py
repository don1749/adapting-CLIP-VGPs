import json
import os
import warnings
import csv
import numpy as np
# from utils.zsg_data import FlickrDataset
# from models.slic_vit import SLICViT
# from models.resnet_high_res import ResNetHighRes
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.vgp_data import FlickrVGPsDataset
from models.vgp_vit import VGPViT
from models.siamese import SiameseNet
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def contains_non_ascii(s):
    return any(ord(c) > 127 for c in s)
def replace_non_ascii(s):
    return ''.join(c if ord(c) < 128 else '?' for c in s)

# Sample CNN model using PyTorch
class HeatMapsComparator(nn.Module):
    def __init__(self, pretrained_model):
        print('Init CNN comparator')
        super(HeatMapsComparator, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 112 * 112 * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.map_model = pretrained_model.eval()
        self.thres = nn.Parameter(torch.tensor([0.5]))
    
    def predict(self, x):
        return x > self.thres

    def forward(self, img, phrases):
        # print(type(img), img.shape)
        x1,x2 = self.map_model(img, phrases)
        # Convert NumPy arrays to PyTorch tensors
        x1 = torch.from_numpy(x1).unsqueeze(0).unsqueeze(0).cuda()
        x2 = torch.from_numpy(x2).unsqueeze(0).unsqueeze(0).cuda()
        x1 = self.pool(self.relu(self.conv1(x1)))
        x2 = self.pool(self.relu(self.conv1(x2)))
        x1 = x1.view(-1, 64 * 112 * 112)
        x2 = x2.view(-1, 64 * 112 * 112)
        x = torch.cat((x1, x2), dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def load_heatmap_tensor(gpu, dataroot, img_path, phrase):
    heatmap_path = f'{dataroot}{img_path}/{phrase}.npz'
    heatmap_path = replace_non_ascii(heatmap_path) if contains_non_ascii(heatmap_path) else heatmap_path
    data = np.load(heatmap_path, allow_pickle=True)
    heatmap = np.array([data[f'arr_{i}'] for i in range(len(data))])
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).to(gpu)
    return heatmap_tensor

def setup_gpu(gpu, args):
    print('Setting up GPUs')
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )      
    return rank

def load_dataset(rank, batch_size, args):
    print('Loading dataset')
    train_dataset = FlickrVGPsDataset(data_type='train')
    val_dataset = FlickrVGPsDataset(data_type='val')
    if args.num_samples > 0:
        train_dataset.image_paths = train_dataset.image_paths[:args.num_samples]
        val_dataset.image_paths = val_dataset.image_paths[:args.num_samples//8]
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank,
        shuffle=False
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	val_dataset,
    	num_replicas=args.world_size,
    	rank=rank,
        shuffle=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )
    return train_loader, val_loader

def setup_model(gpu, args):
    # Score map model
    print('Setting up model')
    if args.map_model == 'vgp_vit':
        map_args = {
            'model': 'vit14',
            'alpha': 0.75,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.02,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
        map_model = VGPViT(**map_args).to(gpu)
    # TODO: other baseline models
    else:
        assert False
    
    # Train Similarity CNN
    if args.model == 'siamese':
        sim_net = SiameseNet()
    else:
        sim_net = HeatMapsComparator(map_model)
    model = DDP(sim_net.to(gpu), device_ids=[gpu], find_unused_parameters=True)
    return model

def train(gpu, args):
    rank = setup_gpu(gpu, args)
    train_loader, val_loader = load_dataset(rank=rank,
                                            batch_size=args.batchsize, 
                                            args=args)

    model = setup_model(gpu, args)

    # Train
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    # 損失関数の定義: pair wise loss
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CosineEmbeddingLoss()
    # pairwise loss, contrastive los

    # 最適化手法の定義
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print('Start train session')
    training_history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    # TensorBoard writer
    # writer = SummaryWriter(log_dir='/work/adapting-CLIP-VGPs/train_logs')

    start = datetime.now()
    for epoch in range(args.epochs):
        # print('Epoch {} / {}'.format(epoch + 1, args.epochs))
        # print('--------------------------------------------')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                if args.model == 'siamese':
                    continue
                model.eval()
                dataloader = val_loader

            epoch_loss = 0.0
            epoch_corrects = 0

            for batch in tqdm(dataloader):
                # indices = batch['idx']
                image_paths = batch['image_idx']
                # print(image_indices)
                images = batch['image']
                phrase_pairs = [list(phrase_pair) for phrase_pair in zip(batch['phrases'][0],batch['phrases'][1])]
                labels = batch['label']

                optimizer.zero_grad() # optimizerを初期化
                with torch.set_grad_enabled(phase=='train'):
                    if args.model == 'siamese':
                        dataroot = f'/work/adapting-CLIP-VGPs/data/flickr/heatmaps/{args.split}/'
                        heatmap1_tensors = [load_heatmap_tensor(gpu, dataroot, img_path, phrases[0]) for img_path, phrases in zip(image_paths, phrase_pairs)]
                        heatmap2_tensors = [load_heatmap_tensor(gpu, dataroot, img_path, phrases[1]) for img_path, phrases in zip(image_paths, phrase_pairs)]
                        outputs = [model(heatmap1, heatmap2) for heatmap1, heatmap2 in zip(heatmap1_tensors, heatmap2_tensors)]
                    else:
                        outputs = [model(image, phrases) for image, phrases in zip(images, phrase_pairs)]
                    
                    outputs_tensor = torch.cat(outputs)
                    labels_tensor = labels.float().unsqueeze(1).cuda()
                    
                    loss = criterion(outputs_tensor, labels_tensor)
                    preds = torch.tensor([model.module.predict(output) for output in outputs])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_corrects += torch.sum(preds==labels).item()
        
            epoch_loss = epoch_loss / (len(dataloader.dataset)-1)
            epoch_acc = epoch_corrects / (len(dataloader.dataset)-1)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                training_history['train_loss'].append(epoch_loss)
                training_history['train_acc'].append(epoch_acc)
                
                # Log the training metrics to TensorBoard
                # writer.add_scalar('Train Loss', epoch_loss, epoch)
                # writer.add_scalar('Train Accuracy', epoch_acc, epoch)
            else:
                if args.model == 'siamese':
                    continue
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                training_history['valid_loss'].append(epoch_loss)
                training_history['valid_acc'].append(epoch_acc)

        print('Epoch {} / {} (train) Loss: {:.4f}, Acc: {:.4f}, (val) Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, args.epochs, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1]))


        # チェックポイントの保存
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   },
                   '/work/adapting-CLIP-VGPs/checkpoints/checkpoint{}.pt'.format(epoch + 1))

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    
    # Save the training history after each epoch
    with open('/work/adapting-CLIP-VGPs/checkpoints/training_history.json', 'w') as f:
        json.dump(training_history, f)
    
    # # Close the TensorBoard writer
    # writer.close()

def eval(gpu, args):
    print('Loading test dataset')
    test_loader = DataLoader(
        FlickrVGPsDataset(data_type='test'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # Score map model
    print('Setting up model')
    if args.map_model == 'vgp_vit':
        map_args = {
            'model': 'vit14',
            'alpha': 0.75,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.02,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
        map_model = VGPViT(**map_args).to(gpu)
    # TODO: other baseline models
    else:
        assert False
    
    # Train Similarity CNN
    model = HeatMapsComparator(map_model).to(gpu)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    PATH = '/work/adapting-CLIP-VGPs/checkpoints/checkpoint1.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    # Initialize variables to keep track of evaluation metrics
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Specify the CSV file path
    csv_file_path = '/work/adapting-CLIP-VGPs/similarity_scores.csv'

    with torch.no_grad():
        for batch in tqdm(test_loader):
                indices = batch['idx']
                image_indices = batch['image_idx']
                images = batch['image']
                phrase_pairs = [list(phrase_pair) for phrase_pair in zip(batch['phrases'][0],batch['phrases'][1])]
                labels = batch['label']

                outputs = [model(image, phrases) for image, phrases in zip(images, phrase_pairs)]
                
                # Convert the tensor values to Python floats
                scores = [float(score.item()) for score in outputs]                

                # Write the scores and additional information to the CSV file
                with open(csv_file_path, 'w', newline='') as csv_file:
                    # Create a CSV writer object
                    csv_writer = csv.writer(csv_file)

                    # Write the header
                    csv_writer.writerow(['Index', 'Image_Index', 'Phrase1', 'Phrase2', 'Score'])

                    # Write the data rows
                    data_rows = zip(indices, image_indices, batch['phrases'][0], batch['phrases'][1], scores)
                    csv_writer.writerows(data_rows)

                # Calculate loss (if needed)
                outputs_tensor = torch.cat(outputs)
                labels_tensor = labels.float().unsqueeze(1).to(gpu)

                if criterion is not None:
                    loss = criterion(outputs_tensor, labels_tensor)
                    test_loss += loss.item()

                # Count correct predictions
                preds = torch.tensor([model.predict(output) for output in outputs])
                
                correct_predictions += torch.sum(preds == labels)

                print(correct_predictions)

                # Total number of samples
                total_samples += labels.size(0)

    # Calculate average loss (if needed)
    average_loss = test_loss / len(test_loader)

    # Calculate accuracy
    accuracy = correct_predictions / total_samples

    # Print or log the evaluation results
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_model', type=str, default='vgp_vit')
    parser.add_argument('--model', type=str, default='siamese')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int,
                        default=0)  # 0 to test all samples
    
    # Multi-GPU settings
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=5, type=int, metavar='E',
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=100, type=int, metavar='B',
                    help='batch size')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'              #
    os.environ['MASTER_PORT'] = '12345'                      #
    # Task
    warnings.filterwarnings("ignore")
    if args.split == 'train':
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    elif args.split == 'test':
        eval(6, args)
    else:
        assert False

if __name__ == '__main__':
    main()
