import argparse
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlopsg24.model import NeuralNetwork

DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3
DEFAULT_OUTPUT = Path("models/classifier.pt")


def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    """
    Evaluates model (model) on a partition of the data (given as loader)

    Returns a dictionary (metrics), currently just with accuracy
    """
    metrics = {metric: 0 for metric in ["accuracy", "F1", "precision", "recall", "total"]}
    model.eval()
    counter = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        metrics["accuracy"] += torch.sum(torch.argmax(out, dim=1) == y).item()
        counter += x.shape[0]

    metrics["accuracy"] /= counter
    return metrics


def main() -> None:
    # CLI arguments
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

    # Device selection
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    # Load preprocessed tensors
    x_train_path = args.data_dir / "x_train.pt"
    y_train_path = args.data_dir / "y_train.pt"
    if not (x_train_path.exists() and y_train_path.exists()):
        raise FileNotFoundError(
            f"Missing preprocessing outputs: {x_train_path} and/or {y_train_path}. "
            "Run preprocessing or pass --data-dir to a folder containing x_train.pt/y_train.pt."
        )

    x_train = torch.load(x_train_path, map_location="cpu")
    y_train = torch.load(y_train_path, map_location="cpu").long()

    x_val = torch.load(args.data_dir / "x_val.pt", map_location="cpu")
    y_val = torch.load(args.data_dir / "y_val.pt", map_location="cpu").long()

    x_test = torch.load(args.data_dir / "x_test.pt", map_location="cpu")
    y_test = torch.load(args.data_dir / "y_test.pt", map_location="cpu").long()

    # Build classifier model
    input_dim = int(x_train.shape[1])  # should be 1024
    model = NeuralNetwork()
    model = model.to(device)

    # Training setup
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR)
    train_metrics = {
        metric: list(0 for i in range(args.epochs)) for metric in ["accuracy", "F1", "precision", "recall", "total"]
    }
    val_metrics = {
        metric: list(0 for i in range(args.epochs)) for metric in ["accuracy", "F1", "precision", "recall", "total"]
    }

    logger.info("Started training")
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        counter = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            train_metrics["accuracy"][epoch] += torch.sum(torch.argmax(out, dim=1) == y).item()
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            counter += x.shape[0]

        train_metrics["accuracy"][epoch] /= counter
        val_metric = evaluate(model, val_loader, device)
        for key in val_metric.keys():
            val_metrics[key][epoch] = val_metric[key]

        train_metrics["total"][epoch] = total / max(1, len(loader))
        logger.info(f"epoch={epoch + 1} loss={train_metrics['total'][epoch]:.4f}")

    test_metrics = evaluate(model, test_loader, device)

    # Save checkpoint
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, DEFAULT_OUTPUT)
    logger.info(f"saved_model={DEFAULT_OUTPUT}")
    for name, metrics in zip(["train", "validation", "test"], [train_metrics, val_metrics, test_metrics]):
        logger.debug(f"{name} metrics: {metrics}")

    logger.info(f"Final test accuracy was {test_metrics['accuracy']}")


if __name__ == "__main__":
    main()
