import argparse
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from mlopsg24.model import NeuralNetwork

DEFAULT_OUTPUT = Path("models/classifier.pt")

def confusion_matrix_counts(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute confusion matrix counts without storing all predictions."""

    model.eval()
    with torch.no_grad():
        # Get first batch to determine number of classes and initialize confusion matrix.
        data_iter = iter(loader)
        try:
            x0, y0 = next(data_iter)
        except StopIteration:
            raise ValueError("Empty loader")

        logits0 = model(x0.to(device))
        k = int(logits0.shape[1])
        cm = torch.zeros((k, k), dtype=torch.int64)

        y_true0 = y0.to(torch.long).view(-1).cpu()
        y_pred0 = torch.argmax(logits0, dim=1).to(torch.long).view(-1).cpu()
        cm += torch.bincount(y_true0 * k + y_pred0, minlength=k * k).reshape(k, k)

        # Process remaining batches.
        for x, y in data_iter:
            logits = model(x.to(device))
            y_true = y.to(torch.long).view(-1).cpu()
            y_pred = torch.argmax(logits, dim=1).to(torch.long).view(-1).cpu()
            cm += torch.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)

    return cm


def plot_learning_curves(train_loss: list[float], train_acc: list[float], val_acc: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(train_loss) == 0 or len(train_acc) == 0 or len(val_acc) == 0:
        raise ValueError("Missing learning-curve data")

    epochs = list(range(1, len(train_loss) + 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_loss, label="train loss", color="#1f77b4")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, label="train acc", color="#2ca02c")
    ax2.plot(epochs, val_acc, label="val acc", color="#ff7f0e", linestyle="--")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.0, 1.0)

    lines, labels = [], []
    for a in (ax1, ax2):
        l, lab = a.get_legend_handles_labels()
        line_handles, line_labels = a.get_legend_handles_labels()
        lines += line_handles
        labels += line_labels
    ax1.set_title("Learning curves")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    cm_counts: torch.Tensor,
    out_path: Path,
    normalize: bool,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = cm_counts.to(torch.float32)

    if normalize:
        row_sums = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
        cm = cm / row_sums

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm.numpy(), cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}{' (normalized)' if normalize else ' (counts)'}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    """
    Evaluates model (model) on a partition of the data (given as loader)

    Returns a dictionary (metrics), currently just with accuracy
    """
    metrics = {metric: 0 for metric in ["accuracy", "F1", "precision", "recall", "total"]}
    model.eval()
    counter = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            metrics["accuracy"] += torch.sum(torch.argmax(out, dim=1) == y).item()
            counter += x.shape[0]

    metrics["accuracy"] /= counter
    return metrics


def main() -> None:
    """Main training loop for the neural network classifier.

    Loads preprocessed embeddings and labels, trains a NeuralNetwork model
    on the training set, evaluates on validation and test sets, and saves
    the trained model checkpoint.

    CLI Arguments:
        --data-dir: Path to folder containing preprocessed tensors (x_train.pt, y_train.pt, etc.)
                    Default: "data/processed"
        --epochs: Number of training epochs. Default: 1
        --device: Device to train on ('auto', 'cpu', or 'cuda').
                  'auto' selects cuda if available, else cpu. Default: 'auto'

    Raises:
        FileNotFoundError: If required preprocessed data files (x_train.pt, y_train.pt)
                          are not found in the specified data directory.

    Returns:
        None

    Side Effects:
        - Trains the model and logs metrics at each epoch
        - Saves trained model state_dict to 'models/classifier.pt'
        - Logs final test accuracy
    """
    # CLI arguments
    parser = argparse.ArgumentParser(description="Minimal training loop on precomputed embeddings")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder created by preprocessing (should contain x_train.pt and y_train.pt)",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: auto|cpu|cuda (auto picks cuda if available else cpu)",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Where to write figures when plotting is enabled",
    )
    parser.add_argument(
        "--plot-learning-curves",
        action="store_true",
        help="If set, writes learning curve figure to --fig-dir",
    )
    parser.add_argument(
        "--plot-confusion-matrix",
        action="store_true",
        default=True,
        help="If set, writes confusion matrix figure to --fig-dir"
    )
    parser.add_argument(
        "--cm-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use for the confusion matrix",
    )
    parser.add_argument(
        "--cm-no-normalize",
        action="store_true",
        help="If set, confusion matrix shows raw counts (default: row-normalized)",
    )
    args = parser.parse_args()

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="g24",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    )

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

    x_test = torch.load(args.data_dir / "x_val.pt", map_location="cpu")
    y_test = torch.load(args.data_dir / "y_val.pt", map_location="cpu").long()

    # Build classifier model
    input_dim = int(x_train.shape[1])  # should be 1024
    model = NeuralNetwork()
    model = model.to(device)

    # Training setup
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss = [0.0 for _ in range(args.epochs)]
    train_acc = [0.0 for _ in range(args.epochs)]
    val_acc = [0.0 for _ in range(args.epochs)]

    logger.info("Started training")
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        counter = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            train_acc[epoch] += torch.sum(torch.argmax(out, dim=1) == y).item()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            counter += x.shape[0]
            wandb.log({"train_loss": loss.item()})

            if i % 100 == 0:
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.cpu())})

        train_acc[epoch] /= counter
        wandb.log({"train_accuracy": train_acc[epoch]})
        val_metric = evaluate(model, val_loader, device)
        val_acc[epoch] = float(val_metric["accuracy"])
        wandb.log({"val_accuracy": val_acc[epoch]})

        train_loss[epoch] = total / max(1, len(loader))
        logger.info(f"epoch={epoch + 1} loss={train_loss[epoch]:.4f}")

    # Optional plotting
    if args.plot_learning_curves:
        curves_path = args.fig_dir / "learning_curves.png"
        plot_learning_curves(train_loss=train_loss, train_acc=train_acc, val_acc=val_acc, out_path=curves_path)
        logger.info(f"saved_figure={curves_path}")

    if args.plot_confusion_matrix:
        split = args.cm_split
        if split == "train":
            cm_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=False)
        elif split == "val":
            cm_loader = val_loader
        else:
            x_test = torch.load(args.data_dir / "x_test.pt", map_location="cpu")
            y_test = torch.load(args.data_dir / "y_test.pt", map_location="cpu").long()
            cm_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

        cm_counts = confusion_matrix_counts(model, cm_loader, device)
        cm_path = args.fig_dir / f"confusion_matrix_{split}.png"
        plot_confusion_matrix(
            cm_counts=cm_counts,
            out_path=cm_path,
            normalize=not args.cm_no_normalize,
            title=f"Confusion matrix ({split})",
        )
        logger.info(f"saved_figure={cm_path}")

    # Save checkpoint
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, DEFAULT_OUTPUT)
    logger.info(f"saved_model={DEFAULT_OUTPUT}")

    test_metric = evaluate(model, test_loader, device)
    final_accuracy = test_metric["accuracy"]

    wandb.log({"final_image": wandb.Image(str(cm_path))})
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="job_classifier",
        type="model",
        description="Job classification model on job texts",
        metadata={"test accuracy": final_accuracy},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

if __name__ == "__main__":
    main()
