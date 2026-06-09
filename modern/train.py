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
  * A fixed, leak-free 90/10 train/val split done once at the file-list level
    (instead of `validation_steps=3`, which validated on ~96 images). Only the
    training pipeline shuffles; the val/test pipelines are deterministic.
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
import random
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
VAL_FRACTION = 0.10
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


def list_examples(root: pathlib.Path, class_to_index: dict[str, int]):
    """Collect (image_path, label) pairs straight from disk."""
    paths, labels = [], []
    for cls, idx in class_to_index.items():
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def split_train_val(paths, labels, val_fraction=VAL_FRACTION):
    """Deterministic, leak-free split done ONCE at the file-list level.

    Splitting here — rather than with take()/skip() on a reshuffling
    tf.data source — guarantees the val set is fixed across epochs and never
    overlaps the training set.
    """
    order = list(range(len(paths)))
    random.Random(SEED).shuffle(order)
    n_val = max(1, int(val_fraction * len(order)))
    val_ids = set(order[:n_val])

    train, val = ([], []), ([], [])
    for i in order:
        bucket = val if i in val_ids else train
        bucket[0].append(paths[i])
        bucket[1].append(labels[i])
    return train, val


def build_pipeline(paths, labels, *, training: bool):
    """Turn (paths, labels) into a batched, prefetched tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Shuffle is BEFORE the decode map, so it buffers cheap (path, label)
    # tuples — not decoded images. Only the training split shuffles.
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.read_file(path)
        # decode_image (not decode_jpeg) handles jpg/png; expand_animations=False
        # keeps it a 3-D tensor so resize/batch work.
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)            # -> float32, 0..255
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])    # pin shape for batching
        return img, label                                # model rescales internally

    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


def main():
    class_names = discover_classes(TRAIN_DIR)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    print(f"{len(class_names)} classes found")

    paths, labels = list_examples(TRAIN_DIR, class_to_index)
    (train_paths, train_labels), (val_paths, val_labels) = split_train_val(paths, labels)
    print(f"{len(train_paths)} train / {len(val_paths)} val images")

    train_ds = build_pipeline(train_paths, train_labels, training=True)
    val_ds = build_pipeline(val_paths, val_labels, training=False)
    test_paths, test_labels = list_examples(TEST_DIR, class_to_index)
    test_ds = build_pipeline(test_paths, test_labels, training=False)

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
