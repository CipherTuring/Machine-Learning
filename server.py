"""Servidor federado.

Etapa 1: los clientes se registran y el servidor mantiene un registro en memoria.
Etapa 2: heartbeats; un monitor marca OFFLINE a los clientes que dejan de responder.
Etapa 3: GET /config para que el cliente arme su partición antes de registrarse.
Etapa 5: una ronda: el servidor publica el modelo global (GET /task) y recibe los pesos
         entrenados de cada cliente (POST /update). Todavía sin agregación.
"""
import threading
import time
from datetime import datetime

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from fl.config import load_config
from fl.model import CNN, count_parameters
from fl.serialization import b64_to_state_dict, describe_state_dict, payload_mb, state_dict_to_b64

CFG = load_config()
HEARTBEAT_TIMEOUT = CFG["federation"]["heartbeat_timeout"]
NUM_CLIENTS = CFG["federation"]["num_clients"]
app = FastAPI(title="Servidor federado")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class ServerState:
    """Estado compartido entre los endpoints HTTP (cada petición puede llegar en otro hilo)."""

    def __init__(self):
        self.lock = threading.Lock()
        # La Condition permite al orquestador "dormir" hasta que llegue un update, sin consumir CPU
        self.cond = threading.Condition(self.lock)
        self.clients = {}       # client_id -> info del cliente
        self.round = 0          # ronda actual (0 = todavía no empezó)
        self.round_open = False
        self.selected = set()   # clientes que participan en la ronda actual
        self.updates = {}       # client_id -> {"n_samples", "state_dict", "metrics"} de la ronda actual
        self.global_b64 = None  # modelo global serializado (se calcula una vez por ronda)
        self.finished = False


STATE = ServerState()


def print_status_table():
    """Tabla con el estado de todos los clientes conocidos (se llama con STATE.lock tomado)."""
    now = time.time()
    lines = ["  Cliente | Estado    | Muestras | Última señal",
             "  --------+-----------+----------+-------------"]
    for cid, c in sorted(STATE.clients.items()):
        lines.append(f"  {cid:>7} | {c['status']:<9} | {c['n_samples']:>8} | "
                     f"hace {now - c['last_seen']:.1f}s")
    print("\n".join(lines), flush=True)


class RegisterRequest(BaseModel):
    client_id: int
    n_samples: int = 0


class HeartbeatRequest(BaseModel):
    client_id: int


class UpdateRequest(BaseModel):
    client_id: int
    round: int
    n_samples: int
    weights: str   # state_dict serializado (base64)
    metrics: dict  # métricas locales del cliente (loss/accuracy de entrenamiento y validación)


def experiment_config():
    return {
        "experiment": CFG["experiment"],
        "training": CFG["training"],
        "federation": CFG["federation"],
    }


@app.get("/config")
def config():
    """El cliente pide la configuración antes de registrarse: necesita el modo/seed para armar su partición."""
    return experiment_config()


@app.post("/register")
def register(req: RegisterRequest):
    with STATE.lock:
        known = req.client_id in STATE.clients
        STATE.clients[req.client_id] = {
            "status": "ONLINE",
            "n_samples": req.n_samples,
            "last_seen": time.time(),
        }
        log(f"Cliente {req.client_id} {'re-registrado' if known else 'registrado'} "
            f"(n_samples={req.n_samples})")
        print_status_table()
    return {"ok": True, **experiment_config()}


@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest):
    with STATE.lock:
        c = STATE.clients.get(req.client_id)
        if c is None:
            # El servidor no conoce a este cliente (p. ej. el servidor se reinició): debe registrarse.
            return {"known": False}
        c["last_seen"] = time.time()
        was_offline = c["status"] == "OFFLINE"
        if was_offline:
            c["status"] = "ONLINE"
            log(f"Cliente {req.client_id} RECONECTADO")
            print_status_table()
    return {"known": True, "was_offline": was_offline}


@app.get("/task")
def task(client_id: int):
    """El cliente pregunta qué hacer: entrenar (recibe el modelo global), esperar o terminar."""
    with STATE.lock:
        if STATE.finished:
            return {"action": "done"}
        c = STATE.clients.get(client_id)
        if (c is None or not STATE.round_open or client_id not in STATE.selected
                or client_id in STATE.updates):
            return {"action": "wait"}
        if c["status"] != "TRAINING":
            c["status"] = "TRAINING"
            log(f"Cliente {client_id} descargó el modelo global de la ronda {STATE.round}")
        return {"action": "train", "round": STATE.round, "weights": STATE.global_b64}


@app.post("/update")
def update(req: UpdateRequest):
    """El cliente devuelve sus pesos entrenados localmente (nunca sus datos)."""
    state_dict = b64_to_state_dict(req.weights)
    with STATE.cond:
        if not STATE.round_open or req.round != STATE.round or req.client_id not in STATE.selected:
            log(f"Update de Cliente {req.client_id} para la ronda {req.round} RECHAZADO "
                f"(ronda actual: {STATE.round})")
            return {"accepted": False}
        STATE.updates[req.client_id] = {
            "n_samples": req.n_samples, "state_dict": state_dict, "metrics": req.metrics}
        STATE.clients[req.client_id]["status"] = "SUBMITTED"
        m = req.metrics
        log(f"Update recibido de Cliente {req.client_id} (ronda {req.round}, n={req.n_samples}, "
            f"{payload_mb(req.weights):.2f} MB): train_loss={m['train_loss']:.4f} "
            f"val_acc={m['val_accuracy']:.4f} val_f1={m['val_f1']:.4f}")
        if len(STATE.updates) == 1:
            log(f"  Contenido del update (solo pesos, sin datos): {describe_state_dict(state_dict)}")
        STATE.cond.notify_all()  # despierta al orquestador
    return {"accepted": True}


@app.get("/status")
def status():
    now = time.time()
    with STATE.lock:
        return {
            cid: {
                "status": c["status"],
                "n_samples": c["n_samples"],
                "last_seen_s_ago": round(now - c["last_seen"], 1),
            }
            for cid, c in sorted(STATE.clients.items())
        }


def monitor_clients():
    """Hilo en segundo plano: un cliente sin heartbeat durante HEARTBEAT_TIMEOUT pasa a OFFLINE."""
    while True:
        time.sleep(1)
        now = time.time()
        with STATE.lock:
            changed = False
            for cid, c in STATE.clients.items():
                silent = now - c["last_seen"]
                if c["status"] != "OFFLINE" and silent > HEARTBEAT_TIMEOUT:
                    c["status"] = "OFFLINE"
                    log(f"Cliente {cid} OFFLINE (sin heartbeat hace {silent:.0f}s)")
                    changed = True
            if changed:
                print_status_table()


def online_clients():
    return sorted(cid for cid, c in STATE.clients.items() if c["status"] != "OFFLINE")


def run_round(r, global_model):
    """Publica el modelo global de la ronda r y espera los updates de los clientes seleccionados."""
    with STATE.cond:
        STATE.round = r
        STATE.selected = set(online_clients())
        STATE.updates = {}
        STATE.global_b64 = state_dict_to_b64(global_model.state_dict())
        STATE.round_open = True
        log(f"===== Ronda {r} iniciada con clientes {sorted(STATE.selected)} "
            f"(modelo global: {payload_mb(STATE.global_b64):.2f} MB) =====")
        print_status_table()
        # Etapa 5: espera a que lleguen todos los updates (en la Etapa 7 se agrega el timeout)
        while not STATE.selected <= STATE.updates.keys():
            STATE.cond.wait(timeout=1)
        STATE.round_open = False
        for cid in STATE.selected:
            STATE.clients[cid]["status"] = "ONLINE"
        log(f"===== Ronda {r} cerrada: updates recibidos de {sorted(STATE.updates)} =====")
        print_status_table()
        return dict(STATE.updates)


def run_federation():
    """Hilo orquestador: espera a los clientes y ejecuta las rondas."""
    torch.manual_seed(CFG["experiment"]["seed"])
    global_model = CNN()
    log(f"Modelo global inicial: CNN con {count_parameters(global_model):,} parámetros")
    log(f"Esperando a que los {NUM_CLIENTS} clientes estén ONLINE...")
    while True:
        with STATE.lock:
            if len(online_clients()) >= NUM_CLIENTS:
                break
        time.sleep(1)
    run_round(1, global_model)
    # Etapa 5: todavía no se agregan los updates (FedAvg llega en la Etapa 6)
    with STATE.lock:
        STATE.finished = True
    log("Federación terminada. Ctrl+C para cerrar el servidor.")


if __name__ == "__main__":
    host, port = CFG["server"]["host"], CFG["server"]["port"]
    threading.Thread(target=monitor_clients, daemon=True).start()
    threading.Thread(target=run_federation, daemon=True).start()
    log(f"Servidor escuchando en {host}:{port}  (modo={CFG['experiment']['mode']}, "
        f"heartbeat_timeout={HEARTBEAT_TIMEOUT}s)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
