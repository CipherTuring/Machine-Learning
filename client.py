"""Cliente federado. Todas las PCs cliente ejecutan este mismo archivo con distinto --id.

Etapa 1: el cliente se registra en el servidor y muestra la configuración recibida.
Etapa 2: hilo de heartbeat ("sigo vivo") y reintentos si no hay conexión.
Etapa 3: el cliente arma su partición local de MNIST según el modo que indica el servidor.
Etapa 4: modelo CNN, entrenamiento local y métricas; modo --local-only (sin servidor).
Etapa 5: bucle de rondas: pedir tarea, entrenar el modelo global recibido y devolver solo los pesos.
Etapa 7: reintentos al enviar pesos y sincronización con el modelo global al reconectarse.
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
from fl.serialization import b64_to_state_dict, payload_mb, state_dict_to_b64
from fl.train import train_local

HTTP_TIMEOUT = 5       # segundos máximos por petición: nunca quedarse colgado esperando al servidor
TRANSFER_TIMEOUT = 30  # peticiones que llevan los pesos del modelo (~2 MB)
POLL_INTERVAL = 1      # cada cuánto pregunta al servidor si hay una ronda nueva
UPDATE_RETRIES = 3     # intentos para enviar los pesos si la red falla


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class FLClient:
    def __init__(self, client_id, server):
        self.id = client_id
        self.server = server.rstrip("/")
        self.cfg = None
        self.n_samples = 0
        self.connected = False
        self.needs_sync = False  # lo activa el hilo de heartbeat al reconectarse
        self.model = CNN()

    def post(self, path, payload, timeout=HTTP_TIMEOUT):
        resp = requests.post(f"{self.server}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get(self, path, timeout=HTTP_TIMEOUT):
        resp = requests.get(f"{self.server}{path}", timeout=timeout)
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
                    self.needs_sync = True
                elif r["was_offline"]:
                    log("El servidor me tenía como OFFLINE: ahora estoy ONLINE otra vez")
                    self.needs_sync = True  # el hilo principal descargará el modelo global vigente
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
        log("Cliente activo. Esperando rondas del servidor (Ctrl+C para salir).")
        while True:
            if self.needs_sync:
                self.sync()
            try:
                task = self.get(f"/task?client_id={self.id}", timeout=TRANSFER_TIMEOUT)
            except requests.RequestException:
                time.sleep(POLL_INTERVAL)  # el hilo de heartbeat ya informa la pérdida de conexión
                continue
            if task["action"] == "done":
                log("El servidor terminó la federación. Fin.")
                return
            if task["action"] == "train":
                self.train_round(task)
            else:
                time.sleep(POLL_INTERVAL)

    def sync(self):
        """Tras una desconexión: descarta el modelo local viejo y descarga el último modelo global."""
        try:
            resp = self.get(f"/model?client_id={self.id}", timeout=TRANSFER_TIMEOUT)
        except requests.RequestException:
            return  # se reintenta en la próxima vuelta del bucle principal
        self.model.load_state_dict(b64_to_state_dict(resp["weights"]))
        self.needs_sync = False
        log(f"SINCRONIZADO con el modelo global de la ronda {resp['round']} | "
            f"validación local: {format_metrics(evaluate(self.model, self.val_loader))}")

    def send_update(self, r, payload):
        """Envía los pesos; si la red falla, reintenta unas pocas veces (el servidor decide si aún sirven)."""
        for attempt in range(1, UPDATE_RETRIES + 1):
            try:
                resp = self.post("/update", payload, timeout=TRANSFER_TIMEOUT)
                if resp["accepted"]:
                    log(f"Ronda {r}: pesos enviados al servidor (aceptados)")
                else:
                    log(f"Ronda {r}: pesos RECHAZADOS por el servidor: {resp['reason']}. "
                        f"Se descartan y se espera la próxima ronda.")
                return
            except requests.RequestException as e:
                log(f"Ronda {r}: no se pudieron enviar los pesos ({type(e).__name__}), "
                    f"intento {attempt}/{UPDATE_RETRIES}")
                time.sleep(2)
        log(f"Ronda {r}: se descartan los pesos de esta ronda (sin conexión con el servidor)")

    def train_round(self, task):
        """Una ronda: cargar modelo global -> entrenar con datos locales -> evaluar -> enviar solo los pesos."""
        r, train_cfg = task["round"], self.cfg["training"]
        self.model.load_state_dict(b64_to_state_dict(task["weights"]))
        log(f"Ronda {r}: modelo global recibido ({payload_mb(task['weights']):.2f} MB). Entrenando...")
        start = time.time()
        stats = train_local(self.model, self.train_loader, train_cfg["local_epochs"],
                            train_cfg["lr"], train_cfg["momentum"])
        val = evaluate(self.model, self.val_loader)
        metrics = {**stats, **{f"val_{k}": v for k, v in val.items()}}
        log(f"Ronda {r}: entrenamiento local listo en {time.time() - start:.1f}s | "
            f"train_loss={stats['train_loss']:.4f}  validación local: {format_metrics(val)}")
        payload = {"client_id": self.id, "round": r, "n_samples": self.n_samples,
                   "weights": state_dict_to_b64(self.model.state_dict()), "metrics": metrics}
        self.send_update(r, payload)


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
