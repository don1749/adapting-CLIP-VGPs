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
from models.text_heatmap_final import TextHeatmapFinal
from utils.heatmap_data import VGPsHeatmapsDataset
from models.HCNN import HCNN
from models.text_heatmap_softmax import TextHeatmapSoftmaxClassifier
from models.text_heatmap_sigmoid import TextHeatmapSigmoidClassifier
from models.text_heatmap_gate import TextHeatmapGatedClassifier
from models.text_map_simple_sigmoid import SimplifiedTextHeatmapGatedClassifier
from models.text_heatmap_sub import TextHeatmapNoCNNSubtraction
from models.text_map_noCNN import TextHeatmapNoCNN
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

def cleanup():
    dist.destroy_process_group()

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
    train_dataset = VGPsHeatmapsDataset(split='train')
    val_dataset = VGPsHeatmapsDataset(split='val')

    if args.num_samples > 0:
        train_dataset.image_idices = train_dataset.image_idices[:args.num_samples]
        val_dataset.image_idices = val_dataset.image_idices[:args.num_samples//28]
    
    # total_size = dataset.__len__()
    # valid_size = int(0.2 * total_size)  # e.g., 20% of the dataset
    # train_size = total_size - valid_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank,
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
    
    # Train Similarity CNN
    if args.model == 'TextHeatmap':
        sim_net = HCNN().to(gpu)
    elif args.model == 'TextHeatmapGatedSigmoid':
        sim_net = TextHeatmapGatedClassifier(heatmap_only=True, text_only=False).to(gpu)
    elif args.model == 'TextHeatmapSoftmax':
        sim_net = TextHeatmapSoftmaxClassifier(heatmap_only=True, use_dropout=True).to(gpu)
    elif args.model == 'TextHeatmapSigmoid':
        sim_net = TextHeatmapSigmoidClassifier(heatmap_only=True).to(gpu)
    elif args.model == 'TextHeatmapNoCNNSigmoid':
        sim_net = TextHeatmapNoCNN(heatmap_only=True).to(gpu)
    elif args.model == 'SimplifiedTextHeatmapSigmoid':
        sim_net = SimplifiedTextHeatmapGatedClassifier(heatmap_only=True).to(gpu)
    elif args.model == 'TextHeatmapNoCNNSubSigmoid':
        sim_net = TextHeatmapNoCNNSubtraction(heatmap_only=True).to(gpu)
    elif args.model == 'TextHeatmapSubCNNSigmoid':
        sim_net = TextHeatmapFinal(heatmap_only=True).to(gpu)
    else:
        assert False

    optimizer = optim.SGD(sim_net.parameters(), lr=args.lr, momentum=0.9)

    if args.checkpoint != 0:
        checkpoint_path = f'/work/adapting-CLIP-VGPs/checkpoints/heatmap only/{args.expno}_checkpoint{args.checkpoint}.pt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
        # Adjust for 'DataParallel' consistency
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            prefix = 'module.'
            name = k[len(prefix):]  # remove `module.` prefixπ
            new_state_dict[name] = v

        sim_net.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = DDP(sim_net, device_ids=[gpu], find_unused_parameters=True)
    return model, optimizer

def train(gpu, args):
    rank = setup_gpu(gpu, args)
    train_loader, val_loader = load_dataset(rank=rank,
                                            batch_size=args.batchsize, 
                                            args=args)

    model, optimizer = setup_model(gpu, args)

    # Train
    train_loss = []
    train_acc = []
    train_prec = []
    train_rec = []
    train_f1 = []
    valid_loss = []
    valid_acc = []
    valid_prec = []
    valid_rec = []
    valid_f1 = []

    history_file_path = '/work/adapting-CLIP-VGPs/checkpoints/heatmap only/{}_training_history.json'.format(args.expno)
    if os.path.isfile(history_file_path):
        with open(history_file_path, 'r') as file:
            data = json.load(file)
            train_loss = data['train_loss']
            train_acc = data['train_acc']
            train_prec = data['train_prec']
            train_rec = data['train_rec']
            train_f1 = data['train_f1']
            valid_loss = data['valid_loss']
            valid_acc = data['valid_acc']
            valid_prec = data['valid_prec']
            valid_rec = data['valid_rec']
            valid_f1 = data['valid_f1']


    criterion = nn.BCEWithLogitsLoss()
    if args.pos_weight != 1:
        pos_weight = torch.tensor([args.pos_weight]).to(gpu)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if 'Softmax' in args.model:
        criterion = nn.CrossEntropyLoss()
        if args.pos_weight !=1:
            pos_weight = args.pos_weight
            neg_weight = pos_weight/(pos_weight-1)
            weights = torch.tensor([neg_weight, pos_weight]).to(gpu)
            criterion = nn.CrossEntropyLoss(weight=weights)

    if args.lr_schedule:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    print('Start train session')
    training_history = {
        'train_loss': train_loss.copy(),
        'train_acc': train_acc.copy(),
        'train_prec':train_prec.copy(),
        'train_rec': train_rec.copy(),
        'train_f1': train_f1.copy(),
        'valid_loss': valid_loss.copy(),
        'valid_acc': valid_acc.copy(),
        'valid_prec':valid_prec.copy(),
        'valid_rec': valid_rec.copy(),
        'valid_f1': valid_f1.copy()
    }
    start = datetime.now()

    for epoch in range(args.checkpoint, args.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            epoch_loss = 0.0
            epoch_TP = 0
            epoch_FP = 0
            epoch_FN = 0
            epoch_TN = 0
            processed = 0

            for batch in tqdm(dataloader):
                image_paths = batch['img_idx']
                left_text_ft = batch['left_text_emb']
                right_text_ft = batch['right_text_emb']
                left_heatmaps = batch['left_heatmap']
                right_heatmaps = batch['right_heatmap']
                labels = batch['label']
                
                left_tensor = left_heatmaps.unsqueeze(1).to(gpu)
                right_tensor = right_heatmaps.unsqueeze(1).to(gpu)
                left_text_ft = left_text_ft.squeeze(1).float().to(gpu)
                right_text_ft = right_text_ft.squeeze(1).float().to(gpu)
                label_tensor = labels.float().unsqueeze(1).to(gpu)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(left_tensor, right_tensor, left_text_ft, right_text_ft)
                    if 'Softmax' in args.model:
                        loss = criterion(outputs, torch.squeeze(label_tensor.type(torch.long)))
                        _, preds = outputs.max(dim=1)
                        preds = preds.unsqueeze(1)
                    elif 'Sigmoid' in args.model:
                        loss = criterion(outputs, label_tensor)
                        preds = (outputs>0.5).float()
                    else:
                        loss = criterion(outputs, label_tensor)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()

                    epoch_loss += loss.item() * len(image_paths)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_TP += ((preds.squeeze(1) == 1) & (label_tensor.squeeze(1) == 1)).float().sum().item()
                    epoch_FP += ((preds.squeeze(1) == 1) & (label_tensor.squeeze(1) == 0)).float().sum().item()
                    epoch_FN += ((preds.squeeze(1) == 0) & (label_tensor.squeeze(1) == 1)).float().sum().item()
                    epoch_TN += ((preds.squeeze(1) == 0) & (label_tensor.squeeze(1) == 0)).float().sum().item()               
                
                processed += len(image_paths)
            
            epoch_loss = epoch_loss / processed
            epoch_acc = (epoch_TP + epoch_TN) / processed
            epoch_prec = epoch_TP / (epoch_TP + epoch_FP) if (epoch_TP + epoch_FP) > 0 else 0
            epoch_rec = epoch_TP / (epoch_TP + epoch_FN) if (epoch_TP + epoch_FN) > 0 else 0
            epoch_f1 = 2 * epoch_prec * epoch_rec / (epoch_prec + epoch_rec) if (epoch_prec + epoch_rec) > 0 else 0

               
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_prec.append(epoch_prec)
                train_rec.append(epoch_rec)
                train_f1.append(epoch_f1)
                training_history['train_loss'].append(epoch_loss)
                training_history['train_acc'].append(epoch_acc)
                training_history['train_prec'].append(epoch_prec)
                training_history['train_rec'].append(epoch_rec)
                training_history['train_f1'].append(epoch_f1)
                # Log the training metrics to TensorBoard
                # writer.add_scalar('Train Loss', epoch_loss, epoch)
                # writer.add_scalar('Train Accuracy', epoch_acc, epoch)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                valid_prec.append(epoch_prec)
                valid_rec.append(epoch_rec)
                valid_f1.append(epoch_f1)
                training_history['valid_loss'].append(epoch_loss)
                training_history['valid_acc'].append(epoch_acc)
                training_history['valid_prec'].append(epoch_prec)
                training_history['valid_rec'].append(epoch_rec)
                training_history['valid_f1'].append(epoch_f1)                 
        
        print('Epoch {} / {}, Learning rate {}'.format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        print('\t(train) Loss: {:.4f}, Acc: {:.4f}, Prec: {:4f}, Rec: {:4f}, F1: {:4f}'.format(train_loss[-1], train_acc[-1], train_prec[-1], train_rec[-1], train_f1[-1]))
        print('\t(val) Loss: {:.4f}, Acc: {:.4f}, Prec: {:4f}, Rec: {:4f}, F1: {:4f}'.format(valid_loss[-1], valid_acc[-1], valid_prec[-1], valid_rec[-1], valid_f1[-1]))
        
        # チェックポイントの保存
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   },
                   '/work/adapting-CLIP-VGPs/checkpoints/heatmap only/{}_checkpoint{}.pt'.format(args.expno,epoch + 1))    
        # Save the training history after each epoch
        with open(history_file_path.format(args.expno), 'w') as f:
            json.dump(training_history, f) 
        if args.lr_schedule:
            scheduler.step()


    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='HCNN')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int,
                        default=0)  # 0 to test all samples
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--expno', type=str, default='01', help='Experiment code')
    
    # Multi-GPU settings
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    
    # learning settings
    parser.add_argument('--epochs', default=5, type=int, metavar='E',
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=100, type=int, metavar='B',
                    help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
    parser.add_argument('--pos_weight', default=1, type=float, 
                    help='weight of positive examples')
    parser.add_argument('--shuffle', default=True, type=bool, 
                    help='shuffle example') 
    parser.add_argument('--lr_schedule', default=False, type=bool,
                    help='Schedule learning rate reduce rate')
    
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'              #
    os.environ['MASTER_PORT'] = '12320'                      #
    # Task
    warnings.filterwarnings("ignore")
    if args.split == 'train':
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    # elif args.split == 'test':
    #     eval(6, args)
    else:
        assert False

if __name__ == '__main__':
    main()
