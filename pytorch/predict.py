"""Predict fruit classes with the trained PyTorch model.

Mirrors ../modern/predict.py. Loads a state_dict (weights only) — which is also
the SAFE way to load models: it deserializes plain tensors, not arbitrary
pickled code. That matters given where this project started (a requirements.txt
full of malware); never `torch.load` an untrusted full-model pickle.

Usage (from the repo root):
    python pytorch/predict.py                   # defaults to fruits/test_images
    python pytorch/predict.py fruits/test_images
    python pytorch/predict.py path/to/one_fruit.jpg
"""
import json
import pathlib
import sys

import torch
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from model import FruitsCNN

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = HERE / "fruits_360.pt"
LABELS_PATH = HERE / "class_names.json"
IMG_SIZE = (100, 100)

# Same scaling as training: uint8 0..255 -> float 0..1, resized to 100x100.
PREP = transforms.Compose([
    transforms.Resize(IMG_SIZE, antialias=True),
    transforms.ConvertImageDtype(torch.float),
])


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gather_paths(target: pathlib.Path) -> list[pathlib.Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return [target]


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        sys.exit(f"No trained model/labels in {HERE}. Run `python pytorch/train.py` first.")

    device = pick_device()
    class_names = json.loads(LABELS_PATH.read_text())

    model = FruitsCNN(len(class_names))
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.to(device).eval()

    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "fruits" / "test_images"
    paths = gather_paths(target)
    if not paths:
        sys.exit(f"No images found at {target}")

    batch = torch.stack([PREP(read_image(str(p), ImageReadMode.RGB)) for p in paths]).to(device)
    with torch.no_grad():
        probs = model(batch).softmax(dim=1)
    conf, idx = probs.max(dim=1)

    for path, i, c in zip(paths, idx.tolist(), conf.tolist()):
        print(f"{path.name:30s} -> {class_names[i]:25s} ({c:.1%})")


if __name__ == "__main__":
    main()
