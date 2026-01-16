import torch.nn as nn

#Neural Network Architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 22),
        )
    def forward(self, x):
        logits = self.model(x)
        return logits