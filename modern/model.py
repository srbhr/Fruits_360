"""Modern Fruits-360 CNN — Keras 3 functional model (TF 2.16+).

What changed vs the 2019 `fruits.py`:

  * Normalization lives INSIDE the model (`Rescaling`), so training and
    inference can never disagree on pixel scaling again. (In 2019 the net
    trained on raw 0-255 pixels and the notebook only divided by 256 for
    *display* — a mismatch waiting to happen.)
  * Data augmentation is part of the model graph and is only active when the
    model is called with `training=True`. This is the modern, can't-misuse-it
    pattern, and it partly fights Fruits-360's biggest weakness: every image
    is a single fruit on a clean white background, so models overfit to that.
  * `Flatten -> Dense(1024)` was 525K params — HALF the old model. Replacing it
    with `GlobalAveragePooling2D` shrinks the model dramatically and usually
    generalizes better.
  * `BatchNormalization` after each conv (wasn't standard-in-every-tutorial in
    2019, is now).
  * The head outputs raw LOGITS (no softmax). The loss applies softmax
    internally (`from_logits=True`), which is numerically more stable.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(num_classes: int, img_size: tuple[int, int] = (100, 100)) -> keras.Model:
    inputs = keras.Input(shape=(*img_size, 3), name="image")

    # Preprocessing baked into the model.
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Augmentation — automatically a no-op at inference (training=False).
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)

    # Convolutional trunk: Conv -> BatchNorm -> ReLU -> Pool.
    for filters in (16, 32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, name="logits")(x)  # logits, no softmax

    return keras.Model(inputs, outputs, name="fruits360_cnn")


if __name__ == "__main__":
    build_cnn(70).summary()
