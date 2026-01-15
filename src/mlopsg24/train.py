from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TensorPairDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same length, got {x.shape[0]} and {y.shape[0]}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# Standard PyTorch training loop:
# - forward pass
# - compute loss
# - backprop
# - optimizer step
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
# Evaluation, only accuracy so far
def accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def _load_split_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train = torch.load(data_dir / "x_train.pt", map_location="cpu")
    y_train = torch.load(data_dir / "y_train.pt", map_location="cpu")
    x_val = torch.load(data_dir / "x_val.pt", map_location="cpu")
    y_val = torch.load(data_dir / "y_val.pt", map_location="cpu")

    if y_train.dtype != torch.long:
        y_train = y_train.long()
    if y_val.dtype != torch.long:
        y_val = y_val.long()

    return x_train, y_train, x_val, y_val


def _make_model(input_dim: int, num_classes: int, hidden_dim: int | None) -> nn.Module:
    if hidden_dim is None or hidden_dim <= 0:
        return nn.Linear(input_dim, num_classes)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


def _make_modelpy_model(input_dim: int, adapter_dim: int = 784) -> nn.Module:
    """Use mlopsg24.model.NeuralNetwork without editing model.py.

    model.py currently expects input vectors of length 784 and outputs 22 logits.
    For text-embedding inputs (e.g., 384-dim or 1024-dim), we add a learnable
    linear adapter to map to 784.
    """

    from mlopsg24.model import NeuralNetwork

    if input_dim == adapter_dim:
        return NeuralNetwork()
    return nn.Sequential(
        nn.Linear(input_dim, adapter_dim),
        NeuralNetwork(),
    )



# Creates:
# - device (CPU/GPU)
# - dataset + train/val split
# - dataloaders
# - baseline model (linear classifier on embeddings)
# - optimizer + loss
# - training loop with printed metrics
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a simple classifier on precomputed text embeddings")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed_sample"),
        help="Folder containing x_train.pt/y_train.pt/x_val.pt/y_val.pt",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--arch",
        choices=["linear", "mlp", "modelpy"],
        default="linear",
        help="Model architecture: linear, 1-hidden-layer MLP, or NeuralNetwork from mlopsg24/model.py",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=0,
        help="Only used for --arch=mlp (hidden layer size).",
    )
    parser.add_argument(
        "--modelpy-adapter-dim",
        type=int,
        default=784,
        help="Only used for --arch=modelpy: dimension to adapt embeddings to before NeuralNetwork.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/classifier.pt"),
        help="Where to save trained model weights (state_dict)",
    )

    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_val, y_val = _load_split_tensors(args.data_dir)
    input_dim = int(x_train.shape[1])
    num_classes = int(max(int(y_train.max().item()), int(y_val.max().item())) + 1)

    train_loader = DataLoader(
        TensorPairDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorPairDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.arch == "modelpy":
        if num_classes != 22:
            raise ValueError(
                f"--arch=modelpy requires exactly 22 classes (model.py outputs 22), got num_classes={num_classes}."
            )
        model = _make_modelpy_model(input_dim=input_dim, adapter_dim=args.modelpy_adapter_dim).to(device)
    elif args.arch == "mlp":
        hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 256
        model = _make_model(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    else:
        model = _make_model(input_dim=input_dim, num_classes=num_classes, hidden_dim=None).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = accuracy(model, val_loader, device)
        print(f"epoch={epoch} loss={loss:.4f} val_acc={val_acc:.3f}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
        args.output_model,
    )
    print(f"saved_model={args.output_model}")


# Script entrypoint: allows `uv run src/mlopsg24/train.py`
if __name__ == "__main__":
    main()
