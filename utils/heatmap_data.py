from torch.utils.data import Dataset
import os
import os.path as osp
import csv
import numpy as np
import blosc

DATA_ROOT = '/work/adapting-CLIP-VGPs/data'

class VGPsHeatmapsDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, split="train", text_only=False):
        self.data_root = data_root
        self.split = split

        self.image_idices = []
        self.phrase_pairs = []
        self.labels = []
        self.heatmaps = {}
        self.textembs = {}
        self.text_only = text_only
        
        self.anno_csv_path = osp.join(
            self.data_root,
            "flickr/phrases_data/phrase_pair_remove_trivial_match_{}.csv".format(split),
        )
        with open(self.anno_csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # image_path = row["image"] + ".jpg"
                # self.image_paths.append(image_path)
                self.image_idices.append(row["image"])
                phrase1 = row["original_phrase1"]
                phrase2 = row["original_phrase2"]
                self.phrase_pairs.append([phrase1, phrase2])
                self.labels.append(row["ytrue"])
    
    def __len__(self):
        return len(self.image_idices)

    def __getitem__(self, idx):
        img_idx = self.image_idices[idx]
        phrase_pair = self.phrase_pairs[idx]
        label = self.labels[idx]=='True'        
        
        # Load heatmap
        if not self.text_only:
            heatmap1_path = get_heatmap_path(
                    dataroot=f'{self.data_root}/flickr/heatmaps/{self.split}/',
                    img_path=img_idx,
                    phrase=phrase_pair[0]
                )
            heatmap2_path = get_heatmap_path(
                    dataroot=f'{self.data_root}/flickr/heatmaps/{self.split}/',
                    img_path=img_idx,
                    phrase=phrase_pair[1]
                )
            
            if heatmap1_path in self.heatmaps:
                heatmap1 = self.heatmaps[heatmap1_path]
            else:
                try:
                    heatmap1 = load_heatmap(heatmap1_path)
                except:
                    print(heatmap1_path)
                self.heatmaps[heatmap1_path] = heatmap1

            if heatmap2_path in self.heatmaps:
                heatmap2 = self.heatmaps[heatmap2_path]
            else:
                try:
                    heatmap2 = load_heatmap(heatmap2_path)
                except:
                    print(heatmap2_path)
                self.heatmaps[heatmap2_path] = heatmap2
        
        # Load text emb
        if phrase_pair[0] in self.textembs:
            left_text_emb = self.textembs[phrase_pair[0]]
        else:
            try:
                left_text_emb = load_text_emb(self.data_root, phrase_pair[0])
            except:
                print(f"Error loading textemb: {phrase_pair[0]}")
            self.textembs[phrase_pair[0]] = left_text_emb

        if phrase_pair[1] in self.textembs:
            right_text_emb = self.textembs[phrase_pair[1]]
        else:
            try:
                right_text_emb = load_text_emb(self.data_root, phrase_pair[1])
            except:
                print(f"Error loading textemb: {phrase_pair[1]}")
            self.textembs[phrase_pair[1]] = right_text_emb
        
        if not self.text_only:
            data = {
                "img_idx": img_idx,
                "phrases": phrase_pair,
                "left_text_emb": left_text_emb,
                "right_text_emb": right_text_emb,
                "left_heatmap": heatmap1,
                "right_heatmap": heatmap2,
                "label": label
            }
        else:
            data = {
                "img_idx": img_idx,
                "phrases": phrase_pair,
                "left_text_emb": left_text_emb,
                "right_text_emb": right_text_emb,
                "label": label
            }
        return data

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

def load_text_emb(dataroot, phrase):
    text_emb_path = f'{dataroot}/flickr/phrase_embs/{phrase}.npy'
    text_emb_path = replace_non_ascii(text_emb_path) if contains_non_ascii(text_emb_path) else text_emb_path
    return np.load(text_emb_path)