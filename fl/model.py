"""Modelo usado en ambos experimentos (IID y non-IID): una CNN pequeña para MNIST."""
import torch.nn as nn


class CNN(nn.Module):
    """conv(1->32) -> pool -> conv(32->64) -> pool -> fc(3136->128) -> fc(128->10).

    Sin BatchNorm a propósito: así el modelo solo tiene pesos y biases, que es exactamente
    lo que se promedia en FedAvg (BatchNorm agrega estadísticas que complican la agregación).
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
