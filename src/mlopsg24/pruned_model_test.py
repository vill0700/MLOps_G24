import time
from mlopsg24.model import NeuralNetwork
import torch

nn = NeuralNetwork()
nn.load_state_dict(torch.load("models/model.pth"))

nn_pruned = NeuralNetwork()
nn_pruned.load_state_dict(torch.load("models/model_pruned.pth"))

def time_network(model: torch.nn) -> float:
    model.eval()
    # model.compile() <- doesn't work for python 3.12
    tic = time.time()
    for _ in range(100):
        _ = model(torch.randn(512, 1024))
    toc = time.time()
    return toc-tic

for name, model in zip(["Unpruned model", "Pruned model"],[nn, nn_pruned]):
    total_time = time_network(model)
    print(f"{name} ran for {total_time} seconds.")