import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
import clip
from spatial_clip import CLIPMaskedSpatialViT
from spatial_clip import CLIPSpatialResNet

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

class VGPViT(nn.Module):
    def __init__(self, model='vit14', alpha=0.8, n_segments=[10, 50, 100, 200],
                 aggregation='mean', temperature=1., compactness=50,
                 sigma=0, **args):
        super().__init__()
        if model == 'vit14':
            args['patch_size'] = 14
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit16':
            args['patch_size'] = 16
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit32':
            args['patch_size'] = 32
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'RN50':
            self.model = CLIPSpatialResNet(**args)
        elif model == 'RN50x4':
            self.model = CLIPSpatialResNet(**args)
        else:
            raise Exception('Invalid model name: {}'.format(model))
        self.alpha = alpha
        self.n_segments = n_segments
        self.aggregation = aggregation
        self.temperature = temperature
        self.compactness = compactness
        self.sigma = sigma
        self.cur_img = torch.empty((), dtype=torch.uint8)
        self.cur_img_feat = []
        self.cur_img_masks = []
        self.txt_feats = {}
        self.heatmaps = {}

    def get_masks(self, im):
        if torch.equal(im.cpu(), self.cur_img.cpu()):
            return self.cur_img_masks

        masks = []
        # Do SLIC with different number of segments so that it has a hierarchical scale structure
        # This can average out spurious activations that happens sometimes when the segments are too small
        for n in self.n_segments:
            segments_slic = slic(im.cpu().numpy().astype(
                np.float32)/255., n_segments=n, compactness=self.compactness, sigma=self.sigma)
            for i in np.unique(segments_slic):
                mask = segments_slic == i
                masks.append(mask)
        masks = np.stack(masks, 0)
        self.cur_img_masks = masks
        return masks

    def get_mask_scores(self, im, text):
        with torch.no_grad():
            if torch.equal(im.cpu(), self.cur_img.cpu()):
                image_features = self.cur_img_feat
                masks = self.get_masks(im)
                masks = torch.from_numpy(masks.astype(np.bool)).cuda()
            # im is uint8 numpy
            else:
                # h, w = im.shape[:2]
                # im = Image.fromarray(im).convert('RGB')
                masks = self.get_masks(im)
                self.cur_img = im
                masks = torch.from_numpy(masks.astype(np.bool)).cuda()
                im = self.model.preprocess(
                        Image.fromarray(im.cpu().numpy())
                    ).unsqueeze(0).cuda()

                image_features = self.model(im, masks)
                image_features = image_features.permute(0, 2, 1)
                image_features = image_features / \
                    image_features.norm(dim=1, keepdim=True)
                self.cur_img_feat = image_features
            
            if text in self.txt_feats:
                text_features = self.txt_feats[text]
            else:
                orig_txt = text
                text = clip.tokenize([text]).cuda()
                text_features = self.model.encode_text(text)
                text_features = text_features / \
                    text_features.norm(dim=1, keepdim=True)
                self.txt_feats[orig_txt] = text_features

            logits = (image_features * text_features.unsqueeze(-1)).sum(1)
            assert logits.size(0) == 1
            logits = logits.cpu().float().numpy()[0]

        return masks.cpu().numpy(), logits

    def get_heatmap(self, im, text):
        if not torch.equal(im.cpu(), self.cur_img.cpu()):
            self.heatmaps = {}
        if text in self.heatmaps:
            return self.heatmaps[text]
        
        masks, logits = self.get_mask_scores(im, text)
        heatmap = list(np.nan + np.zeros(masks.shape, dtype=np.float32))
        for i in range(len(masks)):
            mask = masks[i]
            score = logits[i]
            heatmap[i][mask] = score
        heatmap = np.stack(heatmap, 0)

        heatmap = np.exp(heatmap / self.temperature)

        if self.aggregation == 'mean':
            heatmap = np.nanmean(heatmap, 0)
        elif self.aggregation == 'median':
            heatmap = np.nanmedian(heatmap, 0)
        elif self.aggregation == 'max':
            heatmap = np.nanmax(heatmap, 0)
        elif self.aggregation == 'min':
            heatmap = -np.nanmin(heatmap, 0)
        else:
            assert False

        mask_valid = np.logical_not(np.isnan(heatmap))
        _min = heatmap[mask_valid].min()
        _max = heatmap[mask_valid].max()
        heatmap[mask_valid] = (heatmap[mask_valid] -
                               _min) / (_max - _min + 1e-8)
        heatmap[np.logical_not(mask_valid)] = 0.
        self.heatmaps[text] = heatmap
        return heatmap

    def forward(self, im, phrases, **args):
        # temporary override paramters in init
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
        # forward
        heatmaps = [self.get_heatmap(im, text) for text in phrases]
        alpha = self.alpha
        # restore paramters
        for key in args:
            setattr(self, key, _args[key])
        return heatmaps
