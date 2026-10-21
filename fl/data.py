"""Carga de MNIST y partición de los datos entre clientes (IID y non-IID).

La partición es determinista: con el mismo seed, cada PC calcula exactamente la misma
división, así que cada cliente puede quedarse solo con su parte sin que nadie le envíe datos.
"""
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NUM_CLASSES = 10

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # media y desviación estándar de MNIST
])


def load_mnist(train=True):
    """Descarga MNIST la primera vez (en ./data) y lo devuelve como Dataset de PyTorch."""
    return datasets.MNIST(DATA_DIR, train=train, download=True, transform=TRANSFORM)


def partition_iid(labels, num_clients, seed):
    """IID: se baraja todo y se corta en partes iguales; cada parte tiene ~ la misma mezcla de clases."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(len(labels)), num_clients)


def partition_dirichlet(labels, num_clients, alpha, seed, min_samples=1000):
    """non-IID: para cada clase, se reparte entre clientes según proporciones ~ Dirichlet(alpha).

    alpha pequeño -> proporciones muy desiguales (cada cliente domina unas pocas clases).
    Se repite el sorteo si algún cliente queda con menos de min_samples (sigue siendo determinista).
    """
    rng = np.random.default_rng(seed)
    while True:
        parts = [[] for _ in range(num_clients)]
        for c in range(NUM_CLASSES):
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_samples:
            return [rng.permutation(p) for p in parts]


def partition(labels, mode, num_clients, alpha, seed):
    if mode == "iid":
        return partition_iid(labels, num_clients, seed)
    if mode == "noniid":
        return partition_dirichlet(labels, num_clients, alpha, seed)
    raise ValueError(f"Modo desconocido: {mode!r} (usar 'iid' o 'noniid')")


def get_client_indices(client_id, labels, experiment_cfg, num_clients):
    """Índices de las muestras de entrenamiento que le tocan a client_id (1..num_clients)."""
    parts = partition(labels, experiment_cfg["mode"], num_clients,
                      experiment_cfg["dirichlet_alpha"], experiment_cfg["seed"])
    return parts[client_id - 1]


def class_counts(labels, indices):
    return np.bincount(labels[indices], minlength=NUM_CLASSES)
