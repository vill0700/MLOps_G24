import torch
import torch.nn as nn

class JobAdClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 768)
        return self.net(x)  # logits: (batch, 20)
