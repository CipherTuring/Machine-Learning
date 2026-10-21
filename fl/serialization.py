"""Conversión de los pesos del modelo (state_dict) a texto para enviarlos por HTTP en JSON, y de vuelta.

Lo único que viaja por la red son estos tensores (pesos y biases de cada capa): nunca imágenes ni etiquetas.
"""
import base64
import io

import torch


def state_dict_to_b64(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_state_dict(text):
    buf = io.BytesIO(base64.b64decode(text))
    # weights_only=True: solo acepta tensores, no ejecuta código arbitrario que alguien pudiera enviar
    return torch.load(buf, weights_only=True)


def payload_mb(text):
    return len(text) / 1e6


def describe_state_dict(state_dict):
    """Resumen legible del contenido: nombre y forma de cada tensor."""
    return ", ".join(f"{name} {list(t.shape)}" for name, t in state_dict.items())
