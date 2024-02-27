import torch
import torch.nn as nn

class TextHeatmapGatedClassifier(nn.Module):
    def __init__(self, heatmap_only=False, text_only=False, use_dropout=False):
        super(TextHeatmapGatedClassifier, self).__init__()
        # 
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.heatmap_only = heatmap_only
        self.text_only = text_only
        self.batchnorm_visual = nn.BatchNorm1d(1000) # bn in visual fusion
        self.batchnorm_text = nn.BatchNorm1d(1000) # bn in language fusion

        # Map CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 10, padding=1),  # 16@216*216
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 16@108*108
            nn.Conv2d(16, 32, 7),
            nn.ReLU(inplace=True),  # 32@102*102
            nn.MaxPool2d(2, stride=2),  # 32@51*51
            
            nn.Conv2d(32, 64, 4),
            nn.ReLU(inplace=True),  # 64@48*48
            nn.MaxPool2d(2, stride=2),  # 64@24*24
        )

        # Visual liners
        self.visual_liner = nn.Sequential(
            nn.Linear(128*24*24, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.visual_fuse = nn.Sequential(
            nn.Linear(1000*2, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.hidden_visual = nn.Sequential(
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
            nn.Tanh()
        )

        # Text liners
        self.text_liner = nn.Sequential(
            nn.Linear(768, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.text_fuse = nn.Sequential(
            nn.Linear(1000*2, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.hidden_text = nn.Sequential(
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
            nn.Tanh()
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
        x = self.visual_liner(x)
        return x

    def visual_fusion(self, v1, v2):
        v = torch.cat((v1, v2), 1)
        v = self.visual_fuse(v)
        return v
    
    def text_forward(self, t):
        t = self.text_liner(t)
        return t
    
    def text_fusion(self, t1, t2):
        t = torch.cat((t1, t2), 1)
        t = self.text_fuse(t)
        return t

    def gate_calc(self, v, t):
        x = torch.cat((v,t), 1)
        g_v = self.visual_gate(x)
        g_t = self.text_gate(x)
        return g_v, g_t

    def forward(self, x1, x2, t1, t2):
        # merge text feature pairs
        t1 = self.text_forward(t1)
        t2 = self.text_forward(t2)
        t = self.text_fusion(t1, t2)
        h_t = self.hidden_text(t)
        if self.text_only:
            logits = self.logits(h_t)
            return logits
        
        # encode image pairs
        v1 = self.visual_forward(x1)
        v2 = self.visual_forward(x2)
        # fuse visual feature pairs
        v = self.visual_fusion(v1, v2)
        h_v = self.hidden_visual(v)
        # Map only
        if self.heatmap_only:
            logits = self.logits(h_v)
            return logits
        
        # fuse visual and text with gates
        visual_gate, text_gate = self.gate_calc(v,t)
        y = visual_gate*h_v + text_gate*h_t

        logits = self.logits(y)
        return logits