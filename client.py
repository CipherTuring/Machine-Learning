"""Cliente federado. Todas las PCs cliente ejecutan este mismo archivo con distinto --id.

Etapa 1: el cliente se registra en el servidor y muestra la configuración recibida.
Etapa 2: hilo de heartbeat ("sigo vivo") y reintentos si no hay conexión.
Etapa 3: el cliente arma su partición local de MNIST según el modo que indica el servidor.
"""
import argparse
import threading
import time
from datetime import datetime

import requests

from fl.data import class_counts, get_client_indices, load_mnist

HTTP_TIMEOUT = 5  # segundos máximos por petición: nunca quedarse colgado esperando al servidor


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class FLClient:
    def __init__(self, client_id, server):
        self.id = client_id
        self.server = server.rstrip("/")
        self.cfg = None
        self.n_samples = 0
        self.connected = False

    def post(self, path, payload):
        resp = requests.post(f"{self.server}{path}", json=payload, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get(self, path):
        resp = requests.get(f"{self.server}{path}", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def with_retry(self, what, fn):
        """Ejecuta fn(); si el servidor no responde, reintenta con espera creciente (1, 2, 4… máx 10 s)."""
        wait = 1
        while True:
            try:
                return fn()
            except requests.RequestException as e:
                log(f"No se pudo {what} ({type(e).__name__}), reintentando en {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, 10)

    def load_data(self):
        """Pide la config al servidor y arma la partición local. Los datos nunca salen de esta PC."""
        self.cfg = self.with_retry("obtener la configuración", lambda: self.get("/config"))
        exp, num_clients = self.cfg["experiment"], self.cfg["federation"]["num_clients"]
        if not 1 <= self.id <= num_clients:
            raise SystemExit(f"--id debe estar entre 1 y {num_clients}")
        self.train_set = load_mnist(train=True)
        labels = self.train_set.targets.numpy()
        self.indices = get_client_indices(self.id, labels, exp, num_clients)
        self.n_samples = len(self.indices)
        counts = class_counts(labels, self.indices)
        log(f"Mis datos (modo={exp['mode']}, seed={exp['seed']}): {self.n_samples} muestras")
        log("  por clase: " + "  ".join(f"{c}:{n}" for c, n in enumerate(counts)))

    def register(self):
        self.cfg = self.with_retry("registrarse", lambda: self.post(
            "/register", {"client_id": self.id, "n_samples": self.n_samples}))
        self.connected = True
        log(f"Registrado en {self.server}. Config: experimento={self.cfg['experiment']}, "
            f"entrenamiento={self.cfg['training']}")

    def heartbeat_loop(self):
        """Hilo en segundo plano: avisa al servidor cada heartbeat_interval segundos que sigue vivo."""
        interval = self.cfg["federation"]["heartbeat_interval"]
        while True:
            time.sleep(interval)
            try:
                r = self.post("/heartbeat", {"client_id": self.id})
                if not self.connected:
                    log("Conexión con el servidor recuperada")
                    self.connected = True
                if not r["known"]:
                    log("El servidor no me conoce (¿se reinició?): registrándome de nuevo")
                    self.register()
                elif r["was_offline"]:
                    log("El servidor me tenía como OFFLINE: ahora estoy ONLINE otra vez")
            except requests.RequestException as e:
                if self.connected:
                    log(f"Sin conexión con el servidor ({type(e).__name__}), reintentando...")
                    self.connected = False

    def run(self):
        self.load_data()
        self.register()
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        log("Cliente activo. Enviando heartbeats (Ctrl+C para salir).")
        while True:  # en etapas siguientes aquí irá el bucle de rondas federadas
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Número de cliente (1, 2, 3)")
    parser.add_argument("--server", required=True, help="URL del servidor, ej. http://192.168.1.10:8000")
    args = parser.parse_args()
    try:
        FLClient(args.id, args.server).run()
    except KeyboardInterrupt:
        log("Cliente detenido.")


if __name__ == "__main__":
    main()
