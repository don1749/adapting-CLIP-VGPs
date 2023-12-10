import os
# from utils.zsg_data import FlickrDataset
# from models.slic_vit import SLICViT
# from models.resnet_high_res import ResNetHighRes
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torchtext.data.utils import get_tokenizer
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.vgp_data import FlickrVGPsDataset
from models.vgp_vit import VGPViT
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
        self.map_model = pretrained_model
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	val_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
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
    writer = SummaryWriter(log_dir='/work/adapting-CLIP-VGPs/train_logs')

    start = datetime.now()
    for epoch in range(args.epochs):
        print('Epoch {} / {}'.format(epoch + 1, args.epochs))
        print('--------------------------------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            epoch_loss = 0.0
            epoch_corrects = 0

            # TODO: process batches of dataloader
            for batch in tqdm(dataloader):
                indices = batch['idx']
                images = batch['image']
                phrase_pairs = [list(phrase_pair) for phrase_pair in zip(batch['phrases'][0],batch['phrases'][1])]
                labels = batch['label']

                optimizer.zero_grad() # optimizerを初期化
                with torch.set_grad_enabled(phase=='train'):
                    outputs = [model(image, phrases) for image, phrases in tqdm(zip(images, phrase_pairs))]
                    outputs_tensor = torch.cat(outputs)
                    labels_tensor = labels.float().unsqueeze(1).cuda()
                    
                    loss = criterion(outputs_tensor, labels_tensor)
                    preds = torch.tensor([model.module.predict(output) for output in outputs])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_corrects += torch.sum(preds==labels)
                    print(f'Epoch_loss: {epoch_loss}\tEpoch_corrects: {epoch_corrects}')
        
            epoch_loss = epoch_loss / (len(dataloader.dataset)-1)
            epoch_acc = epoch_corrects / (len(dataloader.dataset)-1)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                training_history['train_loss'].append(epoch_loss)
                training_history['train_acc'].append(epoch_acc)
                
                # Log the training metrics to TensorBoard
                writer.add_scalar('Train Loss', epoch_loss, epoch)
                writer.add_scalar('Train Accuracy', epoch_acc, epoch)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                training_history['valid_loss'].append(epoch_loss)
                training_history['valid_acc'].append(epoch_acc)

                # Log the validation metrics to TensorBoard
                writer.add_scalar('Validation Loss', epoch_loss, epoch)
                writer.add_scalar('Validation Accuracy', epoch_acc, epoch)

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
    
    # Close the TensorBoard writer
    writer.close()

    return train_loss, train_acc, valid_loss, valid_acc

def eval(gpu, args):
    test_loader = DataLoader(
        FlickrVGPsDataset(data_type='test'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    pass

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_model', type=str, default='vit14')
    parser.add_argument('--task', type=str, default='train')
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
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    train(0, args)

    # Task
    if args.task == 'train':
        train(0, args)
    elif args.task == 'test':
        eval(0, args)
    else:
        assert False

if __name__ == '__main__':
    main()
