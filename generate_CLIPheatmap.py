import json
import os
import warnings
import csv
# from utils.zsg_data import FlickrDataset
# from models.slic_vit import SLICViT
# from models.resnet_high_res import ResNetHighRes
import numpy as np
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
from models.slic_vit import SLICViT
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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

def cleanup():
    dist.destroy_process_group()


def load_dataset(rank, batch_size, args):
    print('Loading dataset')
    dataset = FlickrVGPsDataset(data_type=args.split)
    if args.num_samples > 0:
        dataset.image_paths = dataset.image_paths[:args.num_samples]
    
    sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank,
        shuffle=False
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )
    return dataloader

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
        map_model = VGPViT(**map_args)
    # TODO: other baseline models
    elif args.map_model == 'slic_vit':
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
        map_model = SLICViT(**map_args)
    else:
        assert False
    
    # Train Similarity CNN
    model = DDP(map_model.to(gpu), device_ids=[gpu], find_unused_parameters=True)

    return model

def contains_non_ascii(s):
    return any(ord(c) > 127 for c in s)
def replace_non_ascii(s):
    return ''.join(c if ord(c) < 128 else '?' for c in s)

def load_heatmap(path):
    data = np.load(path, allow_pickle=True)
    heatmap = np.array([data[f'arr_{i}'] for i in range(len(data))], dtype=np.float32)
    # heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).to(gpu)
    return heatmap

def save_heatmap(path, heatmap):
    try:
        np.savez_compressed(path, *heatmap)
    except Exception as e:
        print(f"Error saving file: {path}")
        print("Error message:", e)

def generate_heatmap(gpu, args):
    rank = setup_gpu(gpu, args)
    dataloader = load_dataset(rank=rank,
                    batch_size=args.batchsize, 
                    args=args)

    model = setup_model(gpu, args)

    print(f'Start generating {args.split} heatmaps')
    start = datetime.now()

    DATA_ROOT = f'/work/adapting-CLIP-VGPs/data/flickr/heatmaps/{args.split}/'

    for batch in tqdm(dataloader):
        # indices = batch['idx']
        image_indices = batch['image_idx']
        images = batch['image']
        phrase_pairs = [list(phrase_pair) for phrase_pair in zip(batch['phrases'][0],batch['phrases'][1])]

        for image_idx, image, phrases in zip(image_indices, images, phrase_pairs):
            image_path = f'{DATA_ROOT}{image_idx}'
            if not os.path.exists(image_path):
                os.makedirs(image_path, exist_ok=True)
            
            heatmap1_path = f'{image_path}/{phrases[0]}.npz'
            heatmap2_path = f'{image_path}/{phrases[1]}.npz'

            heatmap1_path = replace_non_ascii(heatmap1_path) if contains_non_ascii(heatmap1_path) else heatmap1_path
            heatmap2_path = replace_non_ascii(heatmap2_path) if contains_non_ascii(heatmap2_path) else heatmap2_path

            if os.path.isfile(heatmap1_path):
                try:
                    load_heatmap(heatmap1_path)
                except:
                    heatmap1 = model(image, [phrases[0]])[0]
                    save_heatmap(heatmap1_path, heatmap1)
                    pass
            else:
                heatmap1 = model(image, [phrases[0]])[0]
                save_heatmap(heatmap1_path, heatmap1)

            if os.path.isfile(heatmap2_path):
                try:
                    load_heatmap(heatmap2_path)
                except:
                    heatmap2 = model(image, [phrases[1]])[0]
                    save_heatmap(heatmap2_path, heatmap2)
                    pass
            else:
                heatmap2 = model(image, [phrases[1]])[0]
                save_heatmap(heatmap2_path, heatmap2)
        
    if gpu == 0:
        print("Generation complete in: " + str(datetime.now() - start))
    cleanup()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_model', type=str, default='vit14')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int,
                        default=0)  # 0 to test all samples
    
    # Multi-GPU settings
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--batchsize', default=100, type=int, metavar='B',
                    help='batch size')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'              #
    os.environ['MASTER_PORT'] = '12347'                      #
    # Task
    warnings.filterwarnings("ignore")
    mp.spawn(generate_heatmap, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
