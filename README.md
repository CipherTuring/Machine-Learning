# Aprendizaje Federado: IID vs non-IID (MNIST + CNN)

Sistema de Aprendizaje Federado distribuido en varias computadoras de una red local:
1 servidor (FedAvg) y 3 clientes, cada uno con su propia partición de MNIST.

| Computadora | Rol |
|---|---|
| PC-A | Servidor federado |
| PC-B | Cliente 1 |
| PC-C | Cliente 2 |
| PC-D | Cliente 3 (mientras no exista, corre como 2º proceso en PC-C) |

## 1. Instalación (en cada computadora)

Requisitos: Python 3.10+ y git.

```powershell
git clone https://github.com/CipherTuring/Machine-Learning.git
cd Machine-Learning
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para actualizar el código más adelante: `git pull`.

Descargar MNIST una vez en cada PC (queda en `data/`, que no se sube a git) y ver la distribución de datos:
```powershell
python analysis/distribution.py
```
Genera `results/distribution_iid.csv`, `results/distribution_noniid.csv` y `results/distribution.png`.

## 2. Red y firewall (solo en la PC del servidor)

1. Conectar todas las PCs a la **misma red** (mismo Wi-Fi o switch).
2. Obtener la IP del servidor:
   ```powershell
   ipconfig
   ```
   Buscar `Dirección IPv4` del adaptador Wi-Fi/Ethernet (ej. `192.168.1.10`).
3. Marcar la red como **Privada**: Configuración → Red e Internet → (tu red) → Tipo de perfil de red → Privada.
4. Abrir el puerto 8000 (PowerShell **como administrador**):
   ```powershell
   netsh advfirewall firewall add rule name="FL" dir=in action=allow protocol=TCP localport=8000
   ```

### Prueba de conectividad

En PC-A (servidor):
```powershell
python -m http.server 8000
```
En PC-B y PC-C, abrir en el navegador `http://<IP_SERVIDOR>:8000`.
Si aparece un listado de archivos, la red y el firewall funcionan. Cerrar el servidor con `Ctrl+C`.

## 3. Ejecutar

Activar el entorno en cada PC (`.venv\Scripts\activate`) y luego:

**PC-A (servidor):**
```powershell
python server.py                                  # usa config.yaml (modo y rondas)
python server.py --mode noniid --rounds 5         # sobrescribe modo/rondas sin editar config.yaml
python server.py --mode iid --out results/prueba  # guarda resultados en otra carpeta
```
Resultados (por defecto en `results/<modo>/`, se sobrescriben en cada ejecución):
`rounds.csv` (métricas del modelo global por ronda), `clients.csv` (métricas locales de cada cliente),
`events.log` (registro de eventos) y `global_model.pt`.

Los clientes reciben el modo del servidor: si se cambia el modo, **reiniciar también los clientes**.

**PCs cliente** (mismo archivo, distinto `--id`):
```powershell
python client.py --id 1 --server http://<IP_SERVIDOR>:8000   # PC-B
python client.py --id 2 --server http://<IP_SERVIDOR>:8000   # PC-C
python client.py --id 3 --server http://<IP_SERVIDOR>:8000   # PC-C (2ª terminal) o PC-D
```

Estado de los clientes en el navegador: `http://<IP_SERVIDOR>:8000/status`

**Entrenamiento local sin servidor** (prueba del modelo; usa el `config.yaml` de esa PC):
```powershell
python client.py --id 1 --local-only --epochs 2
```
