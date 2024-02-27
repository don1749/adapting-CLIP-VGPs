import torch
import torch.nn as nn

class TextHeatmapNoCNNSubtraction(nn.Module):
    def __init__(self, heatmap_only=False, text_only=False):
        super(TextHeatmapNoCNNSubtraction, self).__init__()
        # Modal setting
        self.heatmap_only = heatmap_only
        self.text_only = text_only

        # Transform
        self.text_liner = nn.Sequential(
            nn.Linear(768, 4096),
            nn.ReLU(inplace=True)
        )
        self.visual_liner = nn.Sequential(
            nn.Linear(224*224, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(inplace=True),
            nn.Linear(16384, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        self.singlemodal_logits = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

        # Gates
        self.visual_gate = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,300),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,300),
            nn.Sigmoid()
        )

        # Hidden vector
        self.hidden_visual = nn.Sequential(
            nn.Linear(4096, 300),
            nn.BatchNorm1d(300)
        )
        self.hidden_text = nn.Linear(4096, 300)

        # Output
        self.multimodal_logits = nn.Sequential(
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def visual_fuse(self, x1, x2):
        x1 = x1.view(x1.size()[0], -1)
        x1 = self.visual_liner(x1)

        x2 = x2.view(x2.size()[0], -1)
        x2 = self.visual_liner(x2)
        # Fuse
        x = torch.abs(x1-x2)
        return x

    def text_fuse(self, x1, x2):
        x1 = self.text_liner(x1)
        x2 = self.text_liner(x2)
        # Fuse
        x = torch.abs(x1-x2)
        return x
    
    def gate_calc(self, v, t):
        x = torch.cat((v,t), 1)
        g_v = self.visual_gate(x)
        g_t = self.text_gate(x)
        return g_v, g_t
    
    def forward(self, v1, v2, t1, t2):
        # Text
        t = self.text_fuse(t1, t2)
        if self.text_only:
            logits = self.singlemodal_logits(t)
            return logits
        
        # Map
        v = self.visual_fuse(v1, v2)
        if self.heatmap_only:
            logits = self.singlemodal_logits(v)
            return logits
    
        # Fusion
        g_v, g_t = self.gate_calc(v,t)
        h_v = self.hidden_visual(v)
        h_t = self.hidden_text(t)
        tanh = nn.Tanh()
        y = g_v*tanh(h_v) + g_t*tanh(h_t)
        logits = self.multimodal_logits(y)
        return logits



