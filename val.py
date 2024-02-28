import torch
from tqdm import tqdm
from collections import OrderedDict
import argparse
from utils.heatmap_data import VGPsHeatmapsDataset
from torch.utils.data import DataLoader
from models.clip_vgp import VgpClassifier
import pandas as pd



def val(model, heatmap_loader, checkpoint_path, gpu, output_type='sigmoid'):
    print(f'Load model from path: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    GPU = gpu
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        prefix = 'module.'
        name = k[len(prefix):]  # remove `module.` prefix
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(GPU)
    model.eval()

    test_results = []

    with torch.no_grad():
        for batch in tqdm(heatmap_loader):
            image_paths = batch['img_idx']
            phrase_pairs = [list(phrase_pair) for phrase_pair in zip(batch['phrases'][0],batch['phrases'][1])]
            left_text_ft = batch['left_text_emb']
            right_text_ft = batch['right_text_emb']
            left_heatmaps = batch['left_heatmap']
            right_heatmaps = batch['right_heatmap']
            labels = batch['label']
            
            left_heatmaps = left_heatmaps.unsqueeze(1).to(GPU)
            right_heatmaps = right_heatmaps.unsqueeze(1).to(GPU)
            left_text_ft = left_text_ft.squeeze(1).float().to(GPU)
            right_text_ft = right_text_ft.squeeze(1).float().to(GPU)
            label_tensor = labels.float().unsqueeze(1).to(GPU)
            
            outputs = model(left_heatmaps, right_heatmaps, left_text_ft, right_text_ft)
            # loss = criterion(outputs, label_tensor)
            if output_type == 'sigmoid':
                preds = (outputs > 0.5).float()
            elif output_type == 'softmax':
                _, preds = outputs.max(dim=1)
                preds = preds.unsqueeze(1)
            for i in range(len(image_paths)):
                test_results.append([image_paths[i], phrase_pairs[i], outputs[i].item(), preds[i].item(), labels[i].item()])      
    return test_results

def load(heatmap_only, text_only, num_samples=0):
    heatmap_dataset = VGPsHeatmapsDataset(split="test", text_only=text_only)
    batch_size = 100
    if num_samples != 0:
        heatmap_dataset.image_idices = heatmap_dataset.image_idices[:num_samples]

    heatmap_loader = DataLoader(
        dataset=heatmap_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    model = VgpClassifier(heatmap_only=heatmap_only, text_only=text_only)
    return model, heatmap_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expno', type=str)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--heatmaponly', type=bool, default=False)
    parser.add_argument('--textonly', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--pathsuffix', type=str, default='')
    args = parser.parse_args()

    model, heatmap_loader = load(heatmap_only=args.heatmaponly, text_only=args.textonly, num_samples=args.num_samples)

    if args.heatmaponly:
        checkpoint_path = f'/work/adapting-CLIP-VGPs/checkpoints/heatmap only/{args.expno}_checkpoint{args.checkpoint}.pt'
    elif args.textonly:
        checkpoint_path = f'/work/adapting-CLIP-VGPs/checkpoints/text_only/{args.expno}_checkpoint{args.checkpoint}.pt'
    else:
        checkpoint_path = f'/work/adapting-CLIP-VGPs/checkpoints/text+heatmap2/{args.expno}_checkpoint{args.checkpoint}.pt'

    test_results = val(model, heatmap_loader, checkpoint_path, args.gpu)
    data = pd.DataFrame(test_results)
    if args.heatmaponly:
        csv_path = f'/work/adapting-CLIP-VGPs/checkpoints/heatmap only/{args.expno}_checkpoint{args.checkpoint}{args.pathsuffix}.csv'
    elif args.textonly:
        csv_path = f'/work/adapting-CLIP-VGPs/checkpoints/text_only/{args.expno}_checkpoint{args.checkpoint}{args.pathsuffix}.csv'
    else:
        csv_path = f'/work/adapting-CLIP-VGPs/checkpoints/text+heatmap2/{args.expno}_checkpoint{args.checkpoint}{args.pathsuffix}.csv'
    print(f'Writing results to path: {csv_path}')
    data.to_csv(csv_path)


if __name__ == '__main__':
    main()