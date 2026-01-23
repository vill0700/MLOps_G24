import time
from mlopsg24.model import NeuralNetwork
import torch
from torchao.quantization import quantize_
from torchao.quantization.quant_api import int8_weight_only

nn = NeuralNetwork()
nn.load_state_dict(torch.load("models/model.pth", map_location="cpu"))

nn_p = NeuralNetwork()
nn_p.load_state_dict(torch.load("models/model_pruned.pth", map_location="cpu"))

nn_q = NeuralNetwork()
quantize_(nn_q, int8_weight_only())
nn_q.load_state_dict(torch.load("models/model_quantized.pth", map_location="cpu"))

nn_pq = NeuralNetwork()
quantize_(nn_pq, int8_weight_only())
nn_pq.load_state_dict(torch.load("models/model_pruned_quantized.pth", map_location="cpu"))



INPUT = torch.randn(100, 512, 1024)

def time_network(model: torch.nn) -> float:
    with torch.no_grad():
        model.eval()

        for _ in range(100): # warmup
            _ = model(torch.randn(512, 1024))

        # model.compile() # missing some c++ thing
        tic = time.time()
        for i in range(100):
            _ = model(INPUT[i])
        toc = time.time()
        return toc-tic

for name, model in zip(["Unpruned model", "Pruned model", 
                        "Quantized model", "Pruned and quantized model"],
                       [nn, nn_p, nn_q, nn_pq]):
    total_time = time_network(model)
    print(f"{name} ran for {total_time} seconds.")