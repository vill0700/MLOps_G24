import torch
from mlopsg24.model import NeuralNetwork


def test_outdim():
    """Test the MyDataset class."""
    x = torch.randn(16, 1024)
    nn = NeuralNetwork()
    nn.eval()
    out = nn(x)
    assert out.shape[1] == 22
