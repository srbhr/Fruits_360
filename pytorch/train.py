"""Train the Fruits-360 classifier in PyTorch.

The training loop here is the SAME explicit loop you already understand from the
TF GradientTape version — except in PyTorch this isn't a special "plain" mode,
it's just how training is written:

    zero_grad -> forward -> loss -> backward -> step

Mirrors ../modern/train.py:
  * Trains on the whole set, with a fixed, leak-free 90/10 train/val split. The
    val split uses a NON-augmenting transform (same lesson as the TF version).
  * Saves a state_dict (.pt) + class_names.json so inference needs no hand-typed
    label map.

Data: clone Fruits-360 (100x100) under the repo root, e.g.
    git clone https://github.com/fruits-360/fruits-360-100x100
This script auto-detects a folder containing Training/ and Test/ (so either
fruits/ or fruits-360-100x100/ works), or set FRUITS_DIR=/path/to/dataset.

Run from the repo root:  python pytorch/train.py
"""
import json
import os
import pathlib
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import FruitsCNN

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_OUT = HERE / "fruits_360.pt"
LABELS_OUT = HERE / "class_names.json"

IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
VAL_FRACTION = 0.10


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():        # Apple-Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_data_dir() -> pathlib.Path:
    """Locate a folder containing both Training/ and Test/.

    Checks, in order: $FRUITS_DIR, an optional path passed as argv[1], then the
    usual clone locations under the repo root — so it doesn't matter whether you
    cloned into fruits/ or fruits-360-100x100/.
    """
    candidates = []
    if os.environ.get("FRUITS_DIR"):
        candidates.append(pathlib.Path(os.environ["FRUITS_DIR"]))
    if len(sys.argv) > 1:
        candidates.append(pathlib.Path(sys.argv[1]))
    candidates += [ROOT / "fruits", ROOT / "fruits-360-100x100", ROOT / "fruits-360"]

    for c in candidates:
        if (c / "Training").is_dir() and (c / "Test").is_dir():
            return c
    # Last resort: any direct child of the repo that looks like the dataset.
    for c in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if (c / "Training").is_dir() and (c / "Test").is_dir():
            return c

    sys.exit(
        "Dataset not found — need a folder containing Training/ and Test/.\n"
        "Clone it under the repo, e.g.:\n"
        "  git clone https://github.com/fruits-360/fruits-360-100x100\n"
        "then re-run, or:  FRUITS_DIR=/path/to/dataset python pytorch/train.py"
    )


def main():
    data_dir = find_data_dir()
    train_dir, test_dir = data_dir / "Training", data_dir / "Test"
    print(f"data: {train_dir}")

    device = pick_device()
    print(f"device: {device}")

    # Augmentation + scaling live in the transforms (the PyTorch way).
    # ToTensor() scales uint8 0..255 -> float 0..1, so training and inference
    # MUST use the same scaling (predict.py mirrors it). No mean/std normalize,
    # to match the TF version's plain Rescaling(1/255).
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # Two views of the same folder so train can augment while val/test cannot.
    train_full = datasets.ImageFolder(train_dir, transform=train_tf)
    val_full = datasets.ImageFolder(train_dir, transform=eval_tf)
    class_names = train_full.classes
    print(f"{len(class_names)} classes")

    # Fixed, leak-free split: one deterministic permutation, disjoint indices.
    g = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(train_full), generator=g).tolist()
    n_val = max(1, int(VAL_FRACTION * len(order)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    # ImageFolder builds its OWN class->index map from the Test folder. If that
    # disagrees with the training order, test labels would silently misalign and
    # test accuracy would be meaningless. Fail loudly instead of reporting noise.
    if test_ds.classes != class_names:
        sys.exit(
            "Test classes don't match training classes — labels would misalign.\n"
            f"  train: {len(class_names)} classes; test: {len(test_ds.classes)} classes"
        )
    print(f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = FruitsCNN(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()    # expects raw logits + integer labels

    def run_epoch(loader, *, train: bool):
        model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    # Start below 0 so epoch 1 always writes a checkpoint — guarantees MODEL_OUT
    # exists for the final test-set load, even on a degenerate (val_acc==0) run.
    best_val = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        print(
            f"epoch {epoch:2d}/{EPOCHS}  "
            f"loss {train_loss:.3f} acc {train_acc:.3f}  |  "
            f"val_loss {val_loss:.3f} val_acc {val_acc:.3f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), str(MODEL_OUT))
            LABELS_OUT.write_text(json.dumps(class_names, indent=2))
            print(f"  ↳ saved {MODEL_OUT.name} (val_acc {best_val:.3f})")

    # Final test-set evaluation with the best checkpoint.
    model.load_state_dict(torch.load(str(MODEL_OUT), map_location=device))
    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"\nTEST  loss {test_loss:.3f}  acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
