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
