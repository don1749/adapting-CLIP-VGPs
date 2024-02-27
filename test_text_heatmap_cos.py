import clip
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import csv
from utils.vgp_data import FlickrVGPsDataset

DATAROOT = '/work/adapting-CLIP-VGPs/data/flickr/heatmaps/train/'
GPU = 7

def text_cos_sim(model, phrases):
    phrases_tensor = clip.tokenize(phrases).to(GPU)
    text_ft = model.encode_text(phrases_tensor)
    return cosine_similarity(text_ft.cpu().detach())[0,1]

def contains_non_ascii(s):
    return any(ord(c) > 127 for c in s)
def replace_non_ascii(s):
    return ''.join(c if ord(c) < 128 else '?' for c in s)
def get_heatmap_path(dataroot, img_path, phrase):
    heatmap_path = f'{dataroot}{img_path}/{phrase}.npz'
    heatmap_path = replace_non_ascii(heatmap_path) if contains_non_ascii(heatmap_path) else heatmap_path
    return heatmap_path

def load_heatmap(path):
    data = np.load(path, allow_pickle=True)
    heatmap = np.array([data[f'arr_{i}'] for i in range(len(data))], dtype=np.float32)
    # heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).to(gpu)
    return heatmap
def hm_cos_sim(dataroot, img_path, phrases):
    h1_path = get_heatmap_path(dataroot, img_path, phrases[0])
    h2_path = get_heatmap_path(dataroot, img_path, phrases[1])
    h1 = load_heatmap(h1_path)
    h2 = load_heatmap(h2_path)
    return cosine_similarity(h1.reshape(1, -1), h2.reshape(1, -1))[0, 0] 

def eval():
    # print('Setting up model')
    # model, _ = clip.load('ViT-L/14', device=GPU)
    print('Loading dataset')
    test_dataset = FlickrVGPsDataset(data_type='train')
    thres = 0.7
    data_cols = ['img_idx', 'phrases', 'heatmap_sim', 'pred', 'ytrue']

    filename = '/work/adapting-CLIP-VGPs/train_hm_test.csv'
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_cols)

        for data in tqdm(test_dataset):
            img_idx = data['image_idx']
            phrases = data['phrases']
            ytrue = data['label']

            # text_sim = text_cos_sim(model,phrases)
            hm_sim = hm_cos_sim(DATAROOT, img_idx, phrases)
            # score = (text_sim+hm_sim)/2
            pred = hm_sim > thres
            
            phrases = [replace_non_ascii(phrase) if contains_non_ascii(phrase) else phrase for phrase in phrases]
            writer.writerow([img_idx, phrases, hm_sim, pred, ytrue])

eval()