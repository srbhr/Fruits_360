# Fruits-360

A fruit and vegetable image classifier. Hand-built in 2019, rebuilt from scratch in 2026.

Same problem, three eras of the toolchain living in one repo: a Keras relic, a faithful TensorFlow port, and a PyTorch version that reaches **99.0% test accuracy** across 260 classes, trained on a laptop GPU in about 25 minutes.

![test accuracy](https://img.shields.io/badge/test_accuracy-99.0%25-2ea043?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ff6f00?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=flat-square)
![license](https://img.shields.io/badge/license-GPLv3-blue?style=flat-square)

## Why this repo looks the way it does

It began as a self-taught learning project: a convolutional network written by hand, off YouTube, documentation, and a lot of grinding. Six years later its `requirements.txt` had rotted into a supply-chain hazard (a `pip freeze` of 100+ packages, including the `sklearn==0.0` placeholder and an unused `darkflow`), and the Keras APIs it relied on had been removed from TensorFlow.

So instead of overwriting the original, this repo keeps all three generations next to each other. The 2019 to 2026 diff is readable in the code itself.

## Three generations

| Where | Stack | Role |
|-------|-------|------|
| `fruits.py`, `Fruits_Detection.ipynb`, `Fruits_360.h5` | Keras 2, HDF5, `fit_generator` | The 2019 original. Preserved, untouched. |
| [`modern/`](modern/) | TensorFlow 2.21, `tf.data`, `GradientTape` | Faithful port to current TensorFlow. No `ImageDataGenerator`, no `.fit` magic. |
| [`pytorch/`](pytorch/) | PyTorch 2.x, Apple MPS | The go-forward version. Recommended. |

## Results

A compact CNN (four convolutional blocks, batch norm, global average pooling, roughly 149K parameters) on the 100x100 Fruits-360 set:

| Run | Epochs | Test accuracy (45,724 images) |
|-----|-------:|------------------------------:|
| Quick | 5 | 95.5% |
| Full | 15 | **99.0%** |

Trained on Apple Silicon (MPS). No cloud, no CUDA.

## Quickstart (PyTorch)

Requires Python 3.10 or newer. PyTorch has no wheels for 3.14 yet, so use 3.13 if that is your system default.

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r pytorch/requirements.txt

# 2. data (260 classes, ~137k training images)
git clone https://github.com/fruits-360/fruits-360-100x100

# 3. train  ->  writes pytorch/fruits_360.pt + class_names.json
python pytorch/train.py             # 15 epochs by default
EPOCHS=5 python pytorch/train.py    # quick pass

# 4. predict
python pytorch/predict.py fruits/test_images
```

`train.py` locates the dataset automatically whether you cloned it as `fruits-360-100x100/` or `fruits/`, or you can point anywhere with `FRUITS_DIR=/path python pytorch/train.py`.

## The honest part

Fruits-360 is single fruits on clean white backgrounds under studio light. A small CNN memorizes that distribution easily, which is why the accuracy runs so high and the confidences sit near 100%. The same model on a real photo of an apple on a kitchen counter will struggle. As a benchmark this is a strong result; for the real world, the next step is transfer learning on a pretrained backbone.

## Dataset and credit

The Fruits-360 dataset is the work of Horea Mureșan and [Mihai Oltean](https://mihaioltean.github.io/), [*Fruit recognition from images using deep learning*](https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning), Acta Univ. Sapientiae, Informatica, Vol. 10, Issue 1, pp. 26-42, 2018.

Current dataset: [github.com/fruits-360](https://github.com/fruits-360) and [Kaggle](https://www.kaggle.com/datasets/moltean/fruits). 260 classes at the time of writing.

<details>
<summary><strong>The 2019 original (how it worked then)</strong></summary>

<br>

`fruits.py` defines and trains the original Keras `Sequential` CNN. `Fruits_Detection.ipynb` loads the pretrained `Fruits_360.h5` weights and predicts images dropped into `fruits/test_images/`. It targeted TensorFlow 2.0 with Keras 2.3.1.

This code is kept exactly as it shipped. It will not run on a current TensorFlow without the API changes demonstrated in [`modern/`](modern/), and it carries the original (unsafe) dependency list. Treat it as an exhibit, not a starting point. Build on [`pytorch/`](pytorch/).

</details>

## License

[GPL-3.0](LICENSE)
