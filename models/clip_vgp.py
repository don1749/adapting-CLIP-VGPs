import torch
import torch.nn as nn

class VgpClassifier(nn.Module):
    def __init__(self, heatmap_only=False, text_only=False):
        super(VgpClassifier, self).__init__()
        # Modal setting
        self.heatmap_only = heatmap_only
        self.text_only = text_only

        # Text Transform
        self.text_liner = nn.Sequential(
            nn.Linear(768, 4096),
            nn.ReLU(inplace=True)
        )
        self.text_fusion_liner = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True)
        )
        self.hidden_text = nn.Sequential(
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
        )

        self.text_logits = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

        # Map CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 10, padding=1),  # 8@216*216
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),  # 8@54*54
            nn.Conv2d(8, 16, 7),
            nn.ReLU(inplace=True),  # 8@48*48
            nn.MaxPool2d(4, stride=4),  # 16@12*12
        )

        # Visual liners
        self.visual_liner = nn.Sequential(
            nn.Linear(16*12*12, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.visual_fusion_liner = nn.Sequential(
            nn.Linear(1000*2, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.hidden_visual = nn.Sequential(
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
        )

        self.map_logits = nn.Sequential(
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Gates
        self.visual_gate = nn.Sequential(
            nn.Linear(1000*2, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,300),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(1000*2, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,300),
            nn.Sigmoid()
        )
        
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            # elif isinstance(m, nn.Linear):
            #     nn.init.xavier_uniform_(m.weight)

    def visual_forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.visual_liner(x)
        return x

    def visual_fuse(self, x1, x2):
        x1 = x1.view(x1.size()[0], -1)
        x1 = self.visual_liner(x1)

        x2 = x2.view(x2.size()[0], -1)
        x2 = self.visual_liner(x2)
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
        t1 = self.text_liner(t1) # 768->4096
        t2 = self.text_liner(t2) # 768->4096
        t = torch.abs(t1-t2)
        if self.text_only:
            logits = self.text_logits(t) # 4906->1, sigmoid
            return logits
        t = self.text_fusion_liner(t) # 4096 -> 1000
        h_t = self.hidden_text(t) # 1000->300
        
        # Map
        v1 = self.visual_forward(v1) # 224*224->CNN->1000
        v2 = self.visual_forward(v2) # 224*224->CNN->1000
        v = torch.cat((v1, v2), 1) 
        v = self.visual_fusion_liner(v) # 2000->1000
        h_v = self.hidden_visual(v) #1000->300
        if self.heatmap_only:
            logits = self.map_logits(h_v)
            return logits
    
        # Text+map Fusion
        g_v, g_t = self.gate_calc(v,t)
        tanh = nn.Tanh()
        y = g_v*tanh(h_v) + g_t*tanh(h_t)
        logits = self.multimodal_logits(y)
        return logits

