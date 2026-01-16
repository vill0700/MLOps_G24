import torch
from mlopsg24.model import NeuralNetwork
from torch.utils.data import DataLoader, TensorDataset


def test_outdim():
    """Test the MyDataset class."""
    x_train = torch.load("data/processed/x_train.pt", map_location="cpu")
    y_train = torch.load("data/processed/y_train.pt", map_location="cpu").long()

    #Training setup
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
    nn = NeuralNetwork()
    nn.eval()
    for x, _ in loader:
        out = nn(x)
        break
    assert out.shape[1] == 22
