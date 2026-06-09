"""Train the Fruits-360 classifier with a plain-TensorFlow training loop.

"Plain TF" here means: a raw `tf.data` input pipeline (no `ImageDataGenerator`)
and an explicit `tf.GradientTape` loop (no `model.fit`). We still use `tf.keras`
for the layer / optimizer / loss building blocks — in TF 2.x there is no
separate "non-Keras" library for those; Keras *is* TensorFlow's high-level API.
What we're dropping is the high-level *workflow* magic, so you can see the
gradients flow.

Replaces the 2019 `fruits.py`. Notable fixes:

  * Trains on the WHOLE training set each epoch. The old
    `steps_per_epoch=1000 // batch_size` silently used ~1000 of 61,488 images
    per epoch — under 2% of the data.
  * A real 90/10 train/val split instead of `validation_steps=3` (which
    validated on ~96 images).
  * Saves the modern `.keras` format AND a `class_names.json`, so inference
    never relies on a hand-typed 70-entry label dict again.

Dataset: download Fruits-360 and unzip so you have
    fruits/Training/<ClassName>/*.jpg
    fruits/Test/<ClassName>/*.jpg
from https://www.kaggle.com/datasets/moltean/fruits

Run from the repo root:  python modern/train.py
"""
import json
import pathlib
import sys

import tensorflow as tf
from tensorflow import keras

from model import build_cnn

# Paths resolve relative to the repo, not the current working directory.
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN_DIR = ROOT / "fruits" / "Training"
TEST_DIR = ROOT / "fruits" / "Test"
MODEL_OUT = HERE / "fruits_360.keras"
LABELS_OUT = HERE / "class_names.json"

IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def discover_classes(train_dir: pathlib.Path) -> list[str]:
    if not train_dir.exists():
        sys.exit(
            f"Training data not found at '{train_dir}'.\n"
            "Download Fruits-360 from https://www.kaggle.com/datasets/moltean/fruits\n"
            "and unzip it so you have fruits/Training/<Class>/*.jpg"
        )
    # sorted() so class indices are deterministic and reproducible.
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())


def make_dataset(root: pathlib.Path, class_to_index: dict[str, int], *, training: bool):
    """Build a raw tf.data pipeline of (image, label) pairs straight from disk."""
    paths, labels = [], []
    for cls, idx in class_to_index.items():
        for p in (root / cls).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p))
                labels.append(idx)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)   # Fruits-360 images are JPEG
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), label      # 0..255; the model rescales

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, len(paths)


def main():
    class_names = discover_classes(TRAIN_DIR)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    print(f"{len(class_names)} classes found")

    full_train, n_train = make_dataset(TRAIN_DIR, class_to_index, training=True)
    test_ds, _ = make_dataset(TEST_DIR, class_to_index, training=False)

    # 90/10 train/val split on the batched dataset.
    n_batches = n_train // BATCH_SIZE
    val_batches = max(1, int(0.10 * n_batches))
    val_ds = full_train.take(val_batches)
    train_ds = full_train.skip(val_batches)

    model = build_cnn(len(class_names), IMG_SIZE)
    model.summary()

    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = keras.metrics.Mean()
    train_acc = keras.metrics.SparseCategoricalAccuracy()
    val_loss = keras.metrics.Mean()
    val_acc = keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss.update_state(loss)
        train_acc.update_state(labels, logits)

    @tf.function
    def val_step(images, labels):
        logits = model(images, training=False)
        val_loss.update_state(loss_fn(labels, logits))
        val_acc.update_state(labels, logits)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        for m in (train_loss, train_acc, val_loss, val_acc):
            m.reset_state()

        for images, labels in train_ds:
            train_step(images, labels)
        for images, labels in val_ds:
            val_step(images, labels)

        print(
            f"epoch {epoch:2d}/{EPOCHS}  "
            f"loss {train_loss.result():.3f} acc {train_acc.result():.3f}  |  "
            f"val_loss {val_loss.result():.3f} val_acc {val_acc.result():.3f}"
        )

        # Checkpoint the best model by validation accuracy.
        if float(val_acc.result()) > best_val:
            best_val = float(val_acc.result())
            model.save(str(MODEL_OUT))
            LABELS_OUT.write_text(json.dumps(class_names, indent=2))
            print(f"  ↳ saved {MODEL_OUT.name} (val_acc {best_val:.3f})")

    # Final evaluation on the held-out test set using the best checkpoint.
    best = keras.models.load_model(str(MODEL_OUT))
    test_loss = keras.metrics.Mean()
    test_acc = keras.metrics.SparseCategoricalAccuracy()
    for images, labels in test_ds:
        logits = best(images, training=False)
        test_loss.update_state(loss_fn(labels, logits))
        test_acc.update_state(labels, logits)
    print(f"\nTEST  loss {test_loss.result():.3f}  acc {test_acc.result():.3f}")


if __name__ == "__main__":
    main()
