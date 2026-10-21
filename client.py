"""Cliente federado. Todas las PCs cliente ejecutan este mismo archivo con distinto --id.

Etapa 1: el cliente se registra en el servidor y muestra la configuración recibida.
"""
import argparse

import requests

HTTP_TIMEOUT = 5  # segundos máximos por petición: nunca quedarse colgado esperando al servidor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Número de cliente (1, 2, 3)")
    parser.add_argument("--server", required=True, help="URL del servidor, ej. http://192.168.1.10:8000")
    args = parser.parse_args()
    server = args.server.rstrip("/")

    print(f"Cliente {args.id} -> registrándose en {server} ...")
    resp = requests.post(f"{server}/register", json={"client_id": args.id, "n_samples": 0},
                         timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    cfg = resp.json()
    print("Registrado correctamente. Configuración recibida del servidor:")
    print(f"  experimento: {cfg['experiment']}")
    print(f"  entrenamiento: {cfg['training']}")


if __name__ == "__main__":
    main()
