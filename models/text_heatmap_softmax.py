import torch
import torch.nn as nn

class TextHeatmapSoftmaxClassifier(nn.Module):
    def __init__(self, heatmap_only=False, text_only=False, gating=False, use_dropout=False):
        super(TextHeatmapSoftmaxClassifier, self).__init__()
        # setting params
        self.heatmap_only = heatmap_only
        self.gating = gating
        self.text_only = text_only
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10, padding=1),  # 64@216*216
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 64@108*108
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),  # 128@102*102
            nn.MaxPool2d(2, stride=2),  # 128@51*51
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),  # 128@48*48
            nn.MaxPool2d(2, stride=2),  # 128@24*24
        )

        # visual liner
        self.visual_liner = nn.Sequential(
            nn.Linear(128*24*24, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        self.merge_visual_liner = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        
        # text liner
        self.text_liner = nn.Sequential(
            nn.Linear(768, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        self.merge_text_liner = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )

        # fuse liner
        self.fuse_liner = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )

        # output
        self.logits = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 2)
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def visual_forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.visual_liner(x)
        return x

    def forward(self, v1, v2, l1, l2):
        # encode image pairs
        v1 = self.visual_forward(v1)
        v2 = self.visual_forward(v2)

        # merge visual feature pairs
        v = torch.cat((v1,v2), 1)
        v = self.merge_visual_liner(v)

        if self.heatmap_only:
            logits = self.logits(v)
            scores = self.softmax(logits)
            return scores
    
        # merge text feature pairs
        l1 = self.text_liner(l1)
        l2 = self.text_liner(l2)
        l = torch.cat((l1, l2), 1)
        l = self.merge_text_liner(l)

        if self.text_only:
            logits = self.logits(l)
            scores = self.softmax(logits)
            return scores
        
        # fuse visual and language
        y = torch.cat((v,l), 1)
        y = self.fuse_liner(y)

        logits = self.logits(y)
        scores = self.softmax(logits)
        return scores