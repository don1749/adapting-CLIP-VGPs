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
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
# from utils.vgp_data import FlickrVGPsDataset
from utils.heatmap_data import VGPsHeatmapsDataset
# from models.vgp_vit import VGPViT
from models.siamese import SiameseNet
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
    dataset = VGPsHeatmapsDataset(split=args.split)

    if args.num_samples > 0:
        dataset.image_idices = dataset.image_idices[:args.num_samples]
    
    data_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank,
        shuffle=False
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=data_sampler
    )
    return data_loader

def setup_model(gpu, args):
    # Score map model
    print('Setting up model')
    
    # Train Similarity CNN
    if args.model == 'siamese':
        sim_net = SiameseNet()
    else:
        assert False
    
    checkpoint_path = '/work/adapting-CLIP-VGPs/checkpoints/checkpoint10.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Adjust for 'DataParallel' consistency
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        prefix = 'module.'
        name = k[len(prefix):]  # remove `module.` prefix
        new_state_dict[name] = v

    sim_net.load_state_dict(new_state_dict)
    model = DDP(sim_net.to(gpu), device_ids=[gpu])
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001)  # Replace with your actual optimizer and settings
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model


def test(gpu, args, test_results):
    rank = setup_gpu(gpu, args)
    test_loader = load_dataset(rank=rank, batch_size=args.batchsize, args=args)

    model = setup_model(gpu, args)
    model.eval()

    # 損失関数の定義: pair wise loss
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CosineEmbeddingLoss()
    # pairwise loss, contrastive los

    # 最適化手法の定義
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print('Start test session')
    start = datetime.now()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_paths = batch['img_idx']
            phrase_pairs = [list(phrase_pair) for phrase_pair in zip(batch['phrases'][0],batch['phrases'][1])]
            left_heatmaps = batch['left_heatmap']
            right_heatmaps = batch['right_heatmap']
            labels = batch['label']
            
            left_tensor = left_heatmaps.unsqueeze(1).to(gpu)
            right_tensor = right_heatmaps.unsqueeze(1).to(gpu)
            label_tensor = labels.float().unsqueeze(1).to(gpu)
            
            outputs = model(left_tensor, right_tensor)
            # loss = criterion(outputs, label_tensor)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            # corrects = torch.sum(preds.squeeze(1) == label_tensor.squeeze(1)).item()
            # print(f'Batch acc:{corrects/len(image_paths)}')

            for i in range(len(image_paths)):
                test_results.append([image_paths[i], phrase_pairs[i], probs[i].item(), preds[i].item(), labels[i].item()])             
    

    if gpu == 0:
        print("Testing complete in: " + str(datetime.now() - start))
    cleanup()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siamese')
    parser.add_argument('--split', type=str, default='test')
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
    os.environ['MASTER_PORT'] = '12346'                      #
    # Task
    warnings.filterwarnings("ignore")
    if args.split == 'test':
        test_results = []
        mp.spawn(test, nprocs=args.gpus, args=(args, test_results))
        # Save the training history after each epoch
        filename = '/work/adapting-CLIP-VGPs/checkpoints/siamese_test_results.csv'
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_idx', 'phrases', 'sim_score', 'pred', 'gt'])
            writer.writerows(test_results)
    # elif args.split == 'test':
    #     eval(6, args)
    else:
        assert False
    

if __name__ == '__main__':
    main()
