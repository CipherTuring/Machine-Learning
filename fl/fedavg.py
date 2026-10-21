"""FedAvg (McMahan et al., 2017): promedio de los pesos de los clientes, ponderado por su cantidad de datos.

    w_global = sum_k (n_k / n_total) * w_k

Un cliente con más muestras de entrenamiento influye proporcionalmente más en el modelo global.
"""


def fedavg(updates):
    """updates: lista de (n_samples, state_dict). Devuelve el state_dict promedio ponderado."""
    total = sum(n for n, _ in updates)
    weights = [n / total for n, _ in updates]
    first = updates[0][1]
    return {
        name: sum(w * sd[name].float() for w, (_, sd) in zip(weights, updates)).to(first[name].dtype)
        for name in first
    }
