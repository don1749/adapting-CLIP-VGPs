import torch
import torch.nn as nn

class TextHeatmapSigmoidClassifier(nn.Module):
    def __init__(self, heatmap_only=False, text_only=False, gating=False, use_dropout=False):
        super(TextHeatmapSigmoidClassifier, self).__init__()
        # 
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.heatmap_only = heatmap_only
        self.text_only = text_only
        self.gating = gating
        self.batchnorm_visual = nn.BatchNorm1d(1000) # bn in visual fusion
        self.batchnorm_text = nn.BatchNorm1d(1000) # bn in language fusion

        # Map CNN
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

        # Visual liners
        self.visual_liner1 = nn.Sequential(
            nn.Linear(128*24*24, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )
        self.visual_liner2 = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.visual_liner3 = nn.Sequential(
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
            nn.Tanh()
        )

        # Text liners
        self.text_liner1 = nn.Sequential(
            nn.Linear(768, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)
        )
        self.text_liner2 = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        self.text_liner3 = nn.Sequential(
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
            nn.Tanh()
        )

        # output
        self.logits = nn.Sequential(
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
        x = self.visual_liner1(x)
        return x

    def visual_fuse(self, v1, v2):
        v = self.visual_liner2(v1+v2)
        v = self.relu(self.batchnorm_visual(v))
        return v
    
    def text_forward(self, t):
        t = self.text_liner1(t)
        return t
    
    def text_fuse(self, t1, t2):
        t = self.text_liner2(t1+t2)
        # Fuse
        t = self.relu(self.batchnorm_text(t))
        return t

    def forward(self, x1, x2, t1, t2):
        # merge text feature pairs
        t1 = self.text_forward(t1)
        t2 = self.text_forward(t2)
        t = self.text_fuse(t1, t2)
        t = self.text_liner3(t)
        if self.text_only:
            logits = self.logits(t)
            return logits
        
        # encode image pairs
        v1 = self.visual_forward(x1)
        v2 = self.visual_forward(x2)
        # fuse visual feature pairs
        v = self.visual_fuse(v1, v2)
        v = self.visual_liner3(v)
        # Map only
        if self.heatmap_only:
            logits = self.logits(v)
            return logits
        
        # fuse visual and text
        y = 0.5*v + 0.5*t
        logits = self.logits(y)
        return logits