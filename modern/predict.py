"""Predict fruit classes for images using the trained model.

Replaces the inference notebook. Fixes from the 2019 version:

  * Labels come from class_names.json (saved at train time), not a hand-typed
    70-entry dict that has to stay in sync with the training class order.
  * The model runs ONCE over the whole batch. The old notebook re-ran
    `model.predict(...)` inside the per-image plotting loop — full inference on
    every image, once per image (O(n^2)).
  * Uses tf.io / tf.image instead of the removed `keras.preprocessing.image`.

Usage (from the repo root):
    python modern/predict.py                       # defaults to fruits/test_images
    python modern/predict.py fruits/test_images
    python modern/predict.py path/to/one_fruit.jpg
"""
import json
import pathlib
import sys

import tensorflow as tf
from tensorflow import keras

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = HERE / "fruits_360.keras"
LABELS_PATH = HERE / "class_names.json"
IMG_SIZE = (100, 100)


def load_image(path: pathlib.Path) -> tf.Tensor:
    img = tf.io.read_file(str(path))
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32)


def gather_paths(target: pathlib.Path) -> list[pathlib.Path]:
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return [target]


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        sys.exit(f"No trained model/labels in {HERE}. Run `python modern/train.py` first.")

    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "fruits" / "test_images"
    paths = gather_paths(target)
    if not paths:
        sys.exit(f"No images found at {target}")

    class_names = json.loads(LABELS_PATH.read_text())
    model = keras.models.load_model(str(MODEL_PATH))

    batch = tf.stack([load_image(p) for p in paths])
    probs = tf.nn.softmax(model(batch, training=False), axis=-1).numpy()

    for i, path in enumerate(paths):
        idx = int(probs[i].argmax())
        print(f"{path.name:30s} -> {class_names[idx]:25s} ({probs[i][idx]:.1%})")


if __name__ == "__main__":
    main()
