import torch
import torch.nn as nn

class TextHeatmapGatedClassifier(nn.Module):
    def __init__(self, heatmap_only=False):
        super(TextHeatmapGatedClassifier, self).__init__()
        self.heatmap_only = heatmap_only

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
        self.visual_liner = nn.Sequential(nn.Linear(128*24*24, 4096), nn.Sigmoid())
        self.merge_visual_liner = nn.Linear(4096*2, 4096)
        self.text_liner = nn.Sequential(nn.Linear(768, 4096, dtype=torch.float16), nn.Sigmoid())
        self.merge_text_liner = nn.Linear(4096*2, 4096)

        self.lang_gate = nn.Sequential(nn.Linear(4096*2, 4096), nn.Sigmoid())
        self.visual_gate = nn.Sequential(nn.Linear(4096*2, 4096), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(4096, 128),
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
        x = self.visual_liner(x)
        return x

    def forward(self, x1, x2, t1, t2):
        # encode image pairs
        h1 = self.visual_forward(x1)
        h2 = self.visual_forward(x2)

        # merge visual feature pairs
        h = torch.cat((h1,h2), 1)
        h = self.merge_visual_liner(h)

        if self.heatmap_only:
            # print('heatmap only')
            scores = self.out(h)
            return scores

        # merge text feature pairs
        t1 = self.text_liner(t1).squeeze(1)
        t2 = self.text_liner(t2).squeeze(1)
        t = torch.cat((t1,t2), 1).float()
        t = self.merge_text_liner(t)
        fused = torch.cat((h,t), 1)
        
        # Calc gates
        visual_gate = self.visual_gate(fused)
        lang_gate = self.lang_gate(fused)
        y = visual_gate*torch.tanh(h) + lang_gate*torch.tanh(t)

        # score the similarity between the 2 encodings
        scores = self.out(y)
        # print('text learned')
        # return scores (without sigmoid) and use bce_with_logit
        # for increased numerical stability
        return scores