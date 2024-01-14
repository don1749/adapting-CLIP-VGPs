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
from utils.heatmap_data import VGPsHeatmapsDataset
from models.text_hm_cnn import TextHeatmapCNN
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from utils.pytorchtools import EarlyStopping


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
        val_dataset.image_idices = val_dataset.image_idices[:args.num_samples//4]
    
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
    if args.model == 'CNN':
        sim_net = TextHeatmapCNN()
    else:
        assert False

    if args.checkpoint != 0:
        checkpoint_path = f'/work/adapting-CLIP-VGPs/checkpoints/checkpoint{args.checkpoint}.pt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
        # Adjust for 'DataParallel' consistency
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            prefix = 'module.'
            name = k[len(prefix):]  # remove `module.` prefix
            new_state_dict[name] = v

        sim_net.load_state_dict(new_state_dict)

    model = DDP(sim_net.to(gpu), device_ids=[gpu])
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
    train_prec = []
    train_rec = []
    train_f1 = []
    valid_loss = []
    valid_acc = []
    valid_prec = []
    valid_rec = []
    valid_f1 = []


    # 損失関数の定義: pair wise loss
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CosineEmbeddingLoss()
    # pairwise loss, contrastive los

    # 最適化手法の定義
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    early_stopping = EarlyStopping(patience=3, verbose=True, path='checkpoints')

    print('Start train session')
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'train_prec':[],
        'train_rec': [],
        'train_f1': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_prec':[],
        'valid_rec': [],
        'valid_f1': []
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
                
                left_heatmaps = left_heatmaps.unsqueeze(1).to(gpu)
                right_heatmaps = right_heatmaps.unsqueeze(1).to(gpu)
                left_text_ft = left_text_ft.squeeze(1).to(gpu)
                right_text_ft = right_text_ft.squeeze(1).to(gpu)
                label_tensor = labels.float().unsqueeze(1).to(gpu)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(left_heatmaps, right_heatmaps, left_text_ft, right_text_ft)
                    loss = criterion(outputs, label_tensor)
                    epoch_loss += loss.item() * len(image_paths)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

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
        
        print('Epoch {} / {}'.format(epoch + 1, args.epochs))
        print('\t(train) Loss: {:.4f}, Acc: {:.4f}, Prec: {:4f}, Rec: {:4f}, F1: {:4f}'.format(train_loss[-1], train_acc[-1], train_prec[-1], train_rec[-1], train_f1[-1]))
        print('\t(val) Loss: {:.4f}, Acc: {:.4f}, Prec: {:4f}, Rec: {:4f}, F1: {:4f}'.format(valid_loss[-1], valid_acc[-1], valid_prec[-1], valid_rec[-1], valid_f1[-1]))
        
        # チェックポイントの保存
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   },
                   '/work/adapting-CLIP-VGPs/checkpoints/checkpoint{}.pt'.format(epoch + 1)) 
        # Save the training history after each epoch
        with open('/work/adapting-CLIP-VGPs/checkpoints/text+heatmap/training_history.json', 'a') as f:
            json.dump(training_history, f) 

        # Early stop
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopped at epoch#{epoch+1}")
            break


    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int,
                        default=0)  # 0 to test all samples
    parser.add_argument('--checkpoint', type=int, default=0)
    
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
    # elif args.split == 'test':
    #     eval(6, args)
    else:
        assert False

if __name__ == '__main__':
    main()
