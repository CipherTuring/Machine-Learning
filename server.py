"""Servidor federado.

Etapa 1: los clientes se registran y el servidor mantiene un registro en memoria.
"""
import threading
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from fl.config import load_config

CFG = load_config()
app = FastAPI(title="Servidor federado")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class ServerState:
    """Estado compartido entre los endpoints HTTP (cada petición puede llegar en otro hilo)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = {}  # client_id -> info del cliente


STATE = ServerState()


class RegisterRequest(BaseModel):
    client_id: int
    n_samples: int = 0


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
    return {
        "ok": True,
        "experiment": CFG["experiment"],
        "training": CFG["training"],
        "federation": CFG["federation"],
    }


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


if __name__ == "__main__":
    host, port = CFG["server"]["host"], CFG["server"]["port"]
    log(f"Servidor escuchando en {host}:{port}  (modo={CFG['experiment']['mode']})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
