# Fruits-360 — PyTorch port (the go-forward version)

The third generation of this project, and the one to build on:

```
fruits.py + Fruits_Detection.ipynb   2019  Keras 2, .h5, fit_generator   (the relic)
../modern/                           2026  TF 2.21, tf.data, GradientTape (the port)
./ (this folder)                     2026  PyTorch 2.x, MPS               (today's default)
```

We deliberately **don't** carry the old `Fruits_360.h5` weights across — Keras
weights don't load into PyTorch, and the model retrains from scratch anyway.

## Why PyTorch here

- **The loop is the norm, not a mode.** The explicit `zero_grad → forward → loss
  → backward → step` loop in `train.py` is how *all* PyTorch is written — the
  same thing you learned as the TF "GradientTape" loop, minus the ceremony.
- **Apple Silicon.** `torch.device("mps")` is mature for a standard CNN like
  this — smoother than `tensorflow-metal`.
- **Augmentation/scaling live in the data transforms**, not the model — the
  idiomatic split of concerns in PyTorch.

## Files

| File | Does |
|------|------|
| `model.py`   | `FruitsCNN(nn.Module)` — same architecture as `../modern/model.py` |
| `train.py`   | `ImageFolder` + `DataLoader`, MPS-aware explicit training loop |
| `predict.py` | batch inference; loads a `state_dict` (weights-only = safe load) |

## Setup (macOS, Apple Silicon)

Requires Python ≥ 3.9.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r pytorch/requirements.txt
```

If you ever hit "op not implemented for MPS", fall back per-op to CPU without
losing GPU for everything else:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Data

Same dataset as the other versions — unzip Fruits-360 at the repo root:

```
fruits/Training/<ClassName>/*.jpg
fruits/Test/<ClassName>/*.jpg
```

<https://www.kaggle.com/datasets/moltean/fruits>

## Run

```bash
python pytorch/train.py                 # writes fruits_360.pt + class_names.json
python pytorch/predict.py               # predicts fruits/test_images
python pytorch/predict.py some_fruit.jpg
```

## TF version → PyTorch version, line for line

| Concept | `../modern/` (TensorFlow) | here (PyTorch) |
|---|---|---|
| Tensor layout | NHWC `(N,100,100,3)` | NCHW `(N,3,100,100)` |
| Input pipeline | raw `tf.data` | `ImageFolder` + `DataLoader` |
| Rescale / augment | in-model `Rescaling` / `Random*` layers | `transforms` on the data |
| Loss on logits | `SparseCategoricalCrossentropy(from_logits=True)` | `nn.CrossEntropyLoss()` |
| Training step | `tf.GradientTape` + `apply_gradients` | `loss.backward()` + `optimizer.step()` |
| Global pooling | `GlobalAveragePooling2D` | `nn.AdaptiveAvgPool2d(1)` |
| Save format | `.keras` | `state_dict` → `.pt` |
| GPU on Mac | `tensorflow-metal` (flaky) | `mps` device (mature) |
