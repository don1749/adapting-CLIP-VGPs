import os
import warnings
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.vgp_data import FlickrVGPsDataset
import argparse
from tqdm import tqdm
from datetime import datetime
import clip

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
    if args.model == 'vit14':
        model, preprocess = clip.load('ViT-L/14')
    else:
        assert False
    
    # Train Similarity CNN
    model = DDP(model.to(gpu), device_ids=[gpu], find_unused_parameters=True)

    return model

def contains_non_ascii(s):
    return any(ord(c) > 127 for c in s)
def replace_non_ascii(s):
    return ''.join(c if ord(c) < 128 else '?' for c in s)

def generate_emb(gpu, model, phrase):
    token = clip.tokenize(phrase).to(gpu)
    return model.encode_text(token)

def save_text_emb(path, txt_emb):
    text_embeddings_np = txt_emb.cpu().numpy()
    try:
        np.save(path, text_embeddings_np)
    except Exception as e:
        print(f"Error saving file: {path}")
        print("Error message:", e)

def load_text_emb(path):
    try:
        np.load(path)
    except Exception as e:
        print(f"Error loading file: {path}")
        print("Error message:", e)

def generate_text_emb(gpu, args):
    rank = setup_gpu(gpu, args)
    dataloader = load_dataset(rank=rank,
                    batch_size=args.batchsize, 
                    args=args)

    model = setup_model(gpu, args)
    model.eval()

    print(f'Start generating text embeddings')
    start = datetime.now()

    DATA_ROOT = f'/work/adapting-CLIP-VGPs/data/flickr/phrase_embs/'
    with torch.no_grad():
        for batch in tqdm(dataloader):
            phrase_set = set(batch['phrases'][0]).union(set(batch['phrases'][1]))
            
            for phrase in phrase_set:
                emb_path = f'{DATA_ROOT}/{phrase}.npy'
                emb_path = replace_non_ascii(emb_path) if contains_non_ascii(emb_path) else emb_path

                if os.path.isfile(emb_path):
                    try:
                        load_text_emb(emb_path)
                    except:
                        emb = generate_emb(gpu, model.module, phrase)
                        save_text_emb(emb_path, emb)
                        pass
                else:
                    emb = generate_emb(gpu, model.module, phrase)
                    save_text_emb(emb_path, emb)

    if gpu == 0:
        print("Generation complete in: " + str(datetime.now() - start))
    cleanup()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vit14')
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
    mp.spawn(generate_text_emb, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
