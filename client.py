"""Cliente federado. Todas las PCs cliente ejecutan este mismo archivo con distinto --id.

Etapa 1: el cliente se registra en el servidor y muestra la configuración recibida.
Etapa 2: hilo de heartbeat ("sigo vivo") y reintentos si no hay conexión.
Etapa 3: el cliente arma su partición local de MNIST según el modo que indica el servidor.
Etapa 4: modelo CNN, entrenamiento local y métricas; modo --local-only (sin servidor).
"""
import argparse
import threading
import time
from datetime import datetime

import requests
import torch
from torch.utils.data import DataLoader

from fl.config import load_config
from fl.data import class_counts, get_client_indices, load_mnist, to_tensor_dataset, train_val_split
from fl.metrics import evaluate, format_metrics
from fl.model import CNN, count_parameters
from fl.train import train_local

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
        """Arma la partición local según self.cfg. Los datos nunca salen de esta PC."""
        exp, num_clients = self.cfg["experiment"], self.cfg["federation"]["num_clients"]
        train_cfg = self.cfg["training"]
        if not 1 <= self.id <= num_clients:
            raise SystemExit(f"--id debe estar entre 1 y {num_clients}")
        mnist = load_mnist(train=True)
        labels = mnist.targets.numpy()
        indices = get_client_indices(self.id, labels, exp, num_clients)
        counts = class_counts(labels, indices)
        train_idx, val_idx = train_val_split(indices, train_cfg["val_fraction"], exp["seed"] + self.id)
        self.n_samples = len(train_idx)  # el servidor pondera FedAvg con este número
        self.train_loader = DataLoader(to_tensor_dataset(mnist, train_idx),
                                       batch_size=train_cfg["batch_size"], shuffle=True)
        self.val_loader = DataLoader(to_tensor_dataset(mnist, val_idx), batch_size=1000)
        log(f"Mis datos (modo={exp['mode']}, seed={exp['seed']}): {len(indices)} muestras "
            f"-> {len(train_idx)} entrenamiento / {len(val_idx)} validación local")
        log("  por clase: " + "  ".join(f"{c}:{n}" for c, n in enumerate(counts)))

    def run_local(self, epochs):
        """Modo --local-only: entrena solo con los datos locales, sin servidor (prueba del modelo y tiempos)."""
        self.cfg = load_config()
        self.load_data()
        train_cfg = self.cfg["training"]
        torch.manual_seed(self.cfg["experiment"]["seed"])
        model = CNN()
        log(f"Modelo CNN con {count_parameters(model):,} parámetros")
        test_loader = DataLoader(to_tensor_dataset(load_mnist(train=False)), batch_size=1000)
        for epoch in range(1, epochs + 1):
            start = time.time()
            stats = train_local(model, self.train_loader, 1, train_cfg["lr"], train_cfg["momentum"])
            elapsed = time.time() - start
            log(f"Época {epoch}/{epochs}: train_loss={stats['train_loss']:.4f} "
                f"train_acc={stats['train_accuracy']:.4f}  ({elapsed:.1f}s)")
            log(f"  validación local: {format_metrics(evaluate(model, self.val_loader))}")
            log(f"  test MNIST (10 clases completas): {format_metrics(evaluate(model, test_loader))}")

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
        # La config la decide el servidor: el cliente la necesita para armar su partición
        self.cfg = self.with_retry("obtener la configuración", lambda: self.get("/config"))
        self.load_data()
        self.register()
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        log("Cliente activo. Enviando heartbeats (Ctrl+C para salir).")
        while True:  # en etapas siguientes aquí irá el bucle de rondas federadas
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Número de cliente (1, 2, 3)")
    parser.add_argument("--server", help="URL del servidor, ej. http://192.168.1.10:8000")
    parser.add_argument("--local-only", action="store_true",
                        help="Entrenar solo con los datos locales, sin servidor (usa config.yaml local)")
    parser.add_argument("--epochs", type=int, default=3, help="Épocas para --local-only")
    args = parser.parse_args()
    if not args.local_only and not args.server:
        parser.error("--server es obligatorio (salvo con --local-only)")
    try:
        client = FLClient(args.id, args.server or "")
        if args.local_only:
            client.run_local(args.epochs)
        else:
            client.run()
    except KeyboardInterrupt:
        log("Cliente detenido.")


if __name__ == "__main__":
    main()
