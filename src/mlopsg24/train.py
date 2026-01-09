from mlopsg24.model import Model
from mlopsg24.data import MyDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


#Temporary simulated dataset of embeddings + labels
class ToyEmbeddingDataset(Dataset):
    def __init__(self, n: int = 2000, dim: int = 384, num_classes: int = 10, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


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



# Creates:
# - device (CPU/GPU)
# - dataset + train/val split
# - dataloaders
# - baseline model (linear classifier on embeddings)
# - optimizer + loss
# - training loop with printed metrics
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ToyEmbeddingDataset(n=2000, dim=384, num_classes=10, seed=42)
    train_ds, val_ds = random_split(dataset, [1600, 400])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = nn.Linear(384, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 4):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = accuracy(model, val_loader, device)
        print(f"epoch={epoch} loss={loss:.4f} val_acc={val_acc:.3f}")


# Script entrypoint: allows `uv run src/mlopsg24/train.py`
if __name__ == "__main__":
    main()