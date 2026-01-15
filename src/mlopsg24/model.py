import torch
import torch.nn as nn

#Neural Network Architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
        )
    def forward(self, x):
        x = self.model(x)
        return nn.LogSoftmax(dim=1)(x)

if __name__ == "__main__":
    model = NeuralNetwork()
    print(model)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 784)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")