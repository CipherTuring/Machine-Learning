"""Evidencia de la distribución de datos por cliente en IID y non-IID.

Recalcula las particiones con el mismo seed que usan los clientes (no necesita la red) y genera:
  - tabla de conteos clase x cliente (results/distribution_<modo>.csv)
  - divergencia Jensen-Shannon de cada cliente respecto a la distribución global
  - gráfico comparativo (results/distribution.png)

Uso:  python analysis/distribution.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fl.config import load_config
from fl.data import NUM_CLASSES, class_counts, load_mnist, partition

RESULTS = ROOT / "results"


def js_divergence(p, q):
    """Divergencia Jensen-Shannon (base 2): 0 = distribuciones idénticas, 1 = totalmente distintas."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return (kl(p, m) + kl(q, m)) / 2


def distribution_table(labels, mode, cfg):
    exp, num_clients = cfg["experiment"], cfg["federation"]["num_clients"]
    parts = partition(labels, mode, num_clients, exp["dirichlet_alpha"], exp["seed"])
    global_dist = np.bincount(labels, minlength=NUM_CLASSES)
    rows = {}
    for k, idx in enumerate(parts, start=1):
        counts = class_counts(labels, idx)
        rows[f"Cliente {k}"] = list(counts) + [counts.sum(), round(js_divergence(counts, global_dist), 4)]
    columns = [str(c) for c in range(NUM_CLASSES)] + ["total", "JS_vs_global"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def plot(tables):
    fig, axes = plt.subplots(len(tables), 2, figsize=(14, 4.5 * len(tables)))
    width = 0.8 / len(next(iter(tables.values())))
    for row, (mode, df) in enumerate(tables.items()):
        counts = df[[str(c) for c in range(NUM_CLASSES)]]
        # Izquierda: muestras por clase de cada cliente (barras agrupadas)
        ax = axes[row, 0]
        for i, (client, values) in enumerate(counts.iterrows()):
            ax.bar(np.arange(NUM_CLASSES) + i * width, values, width, label=client)
        ax.set_xticks(np.arange(NUM_CLASSES) + width * (len(counts) - 1) / 2, range(NUM_CLASSES))
        ax.set_title(f"{mode.upper()}: muestras por clase y cliente")
        ax.set_xlabel("Clase (dígito)")
        ax.set_ylabel("Muestras")
        ax.set_ylim(0, counts.values.max() * 1.25)  # espacio para la leyenda
        ax.legend(ncol=len(counts), loc="upper center")
        # Derecha: proporción de cada clase dentro de cada cliente (barras apiladas)
        ax = axes[row, 1]
        props = counts.div(counts.sum(axis=1), axis=0)
        bottom = np.zeros(len(props))
        colors = plt.cm.tab10.colors
        for c in range(NUM_CLASSES):
            ax.barh(props.index, props[str(c)], left=bottom, color=colors[c], label=str(c))
            bottom += props[str(c)].values
        ax.set_title(f"{mode.upper()}: composición de clases por cliente")
        ax.set_xlabel("Proporción")
        ax.legend(title="Clase", ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    out = RESULTS / "distribution.png"
    fig.savefig(out, dpi=120)
    print(f"Gráfico guardado en {out}")


def main():
    cfg = load_config()
    labels = load_mnist(train=True).targets.numpy()
    RESULTS.mkdir(exist_ok=True)
    tables = {}
    for mode in ("iid", "noniid"):
        df = distribution_table(labels, mode, cfg)
        tables[mode] = df
        df.to_csv(RESULTS / f"distribution_{mode}.csv")
        extra = f" (Dirichlet alpha={cfg['experiment']['dirichlet_alpha']})" if mode == "noniid" else ""
        print(f"\n=== {mode.upper()}{extra} ===")
        print(df.to_string())
    plot(tables)


if __name__ == "__main__":
    main()
