"""Servidor federado.

Etapa 1: los clientes se registran y el servidor mantiene un registro en memoria.
Etapa 2: heartbeats; un monitor marca OFFLINE a los clientes que dejan de responder.
"""
import threading
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from fl.config import load_config

CFG = load_config()
HEARTBEAT_TIMEOUT = CFG["federation"]["heartbeat_timeout"]
app = FastAPI(title="Servidor federado")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class ServerState:
    """Estado compartido entre los endpoints HTTP (cada petición puede llegar en otro hilo)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = {}  # client_id -> info del cliente


STATE = ServerState()


def print_status_table():
    """Tabla con el estado de todos los clientes conocidos (se llama con STATE.lock tomado)."""
    now = time.time()
    lines = ["  Cliente | Estado   | Muestras | Última señal",
             "  --------+----------+----------+-------------"]
    for cid, c in sorted(STATE.clients.items()):
        lines.append(f"  {cid:>7} | {c['status']:<8} | {c['n_samples']:>8} | "
                     f"hace {now - c['last_seen']:.1f}s")
    print("\n".join(lines), flush=True)


class RegisterRequest(BaseModel):
    client_id: int
    n_samples: int = 0


class HeartbeatRequest(BaseModel):
    client_id: int


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
    return {
        "ok": True,
        "experiment": CFG["experiment"],
        "training": CFG["training"],
        "federation": CFG["federation"],
    }


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


if __name__ == "__main__":
    host, port = CFG["server"]["host"], CFG["server"]["port"]
    threading.Thread(target=monitor_clients, daemon=True).start()
    log(f"Servidor escuchando en {host}:{port}  (modo={CFG['experiment']['mode']}, "
        f"heartbeat_timeout={HEARTBEAT_TIMEOUT}s)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
