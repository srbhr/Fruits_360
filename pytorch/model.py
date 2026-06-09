"""Fruits-360 CNN in PyTorch — the go-forward version (torch 2.x, MPS-ready).

Same architecture as ../modern/model.py (the TF version), translated to a plain
nn.Module. Three idiomatic differences worth noticing:

  * Tensors are channels-FIRST in PyTorch: (N, 3, 100, 100), not (N, 100, 100, 3).
  * Normalization and augmentation do NOT live in the model — in PyTorch they
    belong in the data `transforms` (see train.py). The model is just the
    network. (In Keras we baked Rescaling / Random* into the model graph.)
  * The head outputs raw logits; nn.CrossEntropyLoss applies log-softmax itself,
    so there's no softmax layer here (same idea as TF's from_logits=True).
"""
import torch
from torch import nn


class FruitsCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            # padding=1 with a 3x3 kernel keeps spatial size before the pool
            # (the equivalent of Keras padding="same").
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)           # GlobalAveragePooling2D
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),             # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = FruitsCNN(70)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\n{n_params:,} trainable parameters")
