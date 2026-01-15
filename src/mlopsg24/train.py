import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3
DEFAULT_OUTPUT = Path("models/classifier.pt")


def main() -> None:
    #CLI arguments
    parser = argparse.ArgumentParser(description="Minimal training loop on precomputed embeddings")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder created by preprocessing (should contain x_train.pt and y_train.pt)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: auto|cpu|cuda (auto picks cuda if available else cpu)",
    )
    args = parser.parse_args()

    #Device selection
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    #Load preprocessed tensors
    x_path = args.data_dir / "x_train.pt"
    y_path = args.data_dir / "y_train.pt"
    if not (x_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            f"Missing preprocessing outputs: {x_path} and/or {y_path}. "
            "Run preprocessing or pass --data-dir to a folder containing x_train.pt/y_train.pt."
        )

    x_train = torch.load(x_path, map_location="cpu")
    y_train = torch.load(y_path, map_location="cpu").long()

    #Build classifier model
    from mlopsg24.model import NeuralNetwork

    input_dim = int(x_train.shape[1])
    if input_dim == 784:
        model = NeuralNetwork()
    else:
        model = nn.Sequential(nn.Linear(input_dim, 784), NeuralNetwork())
    model = model.to(device)

    #Training setup
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR)

    #Training loop
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.4f}")

    #Save checkpoint
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, DEFAULT_OUTPUT)
    print(f"saved_model={DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
