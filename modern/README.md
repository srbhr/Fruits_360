# Fruits-360 — modernized pipeline (2026)

A from-scratch rewrite of the 2019 relic in `../fruits.py` + `../Fruits_Detection.ipynb`,
using current TensorFlow (2.16+, which ships Keras 3) with a **raw `tf.data`
input pipeline** and an **explicit `tf.GradientTape` training loop** — no
`ImageDataGenerator`, no `model.fit_generator`, no `.h5`.

> Note: "plain TensorFlow" does **not** mean avoiding `tf.keras.layers`. In TF 2.x,
> Keras *is* TensorFlow's high-level API — layers, optimizers, and losses all live
> there. What we drop is the high-level *workflow* (`ImageDataGenerator` / `.fit`),
> so the data loading and the training step are yours to see and control.

## Files

| File | Replaces | Does |
|------|----------|------|
| `model.py`   | the model block in `fruits.py` | defines the CNN (functional, Keras 3) |
| `train.py`   | `fruits.py`                    | raw `tf.data` pipeline + `GradientTape` loop |
| `predict.py` | `Fruits_Detection.ipynb`       | batch inference on a folder or single image |
| `requirements.txt` | the malware-laden root `requirements.txt` | 4 real deps |

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r modern/requirements.txt
```

The arm64 `tensorflow` wheel runs on the **CPU** out of the box — which is plenty
for this tiny model. GPU (Metal) is optional; uncomment `tensorflow-metal` in
`requirements.txt` only if you want to experiment with it.

## Data

Download Fruits-360 and unzip at the repo root:

```
fruits/Training/<ClassName>/*.jpg
fruits/Test/<ClassName>/*.jpg
```

<https://www.kaggle.com/datasets/moltean/fruits>

## Run

```bash
python modern/train.py                 # trains, writes fruits_360.keras + class_names.json
python modern/predict.py               # predicts fruits/test_images
python modern/predict.py some_fruit.jpg
```

## Will it train on an M2 Pro / 16 GB?

Yes, comfortably. The model is ~200K params and `tf.data` streams images from
disk batch-by-batch, so the dataset never sits in RAM (peak usage is a few
hundred MB). On the CPU expect a few minutes per epoch; 15 epochs is well under
an hour.

## What changed vs 2019

- `ImageDataGenerator.flow_from_directory` → raw `tf.data` (`list` → `from_tensor_slices` → `map(decode/resize)`)
- `model.fit_generator(...)` (removed in TF 2.x) → explicit `GradientTape` loop
- `steps_per_epoch=1000//batch_size` bug (saw <2% of data/epoch) → trains on the full set
- `validation_steps=3` → real 90/10 train/val split
- `Flatten → Dense(1024)` (525K params) → `GlobalAveragePooling2D` (~200K total)
- pixel rescaling + augmentation → in-model `Rescaling` / `Random*` layers
- softmax in the model → logits + `from_logits=True` loss (numerically stable)
- hand-typed 70-label dict → `class_names.json` saved at train time
- `.h5` → native `.keras` format
- `requirements.txt` (100+ pkgs, `sklearn==0.0`, `darkflow`) → 4 real deps
