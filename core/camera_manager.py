"""
core/camera_manager.py
Gestor de Red de Cámaras Domésticas (Fleet Manager) para PawSentry AI.
Permite persistir, sondear, descubrir y alternar cámaras IP (RTSP/HTTP) y Webcams USB.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Asegurar raíz en sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from core.schemas import CameraProfile, CameraProtocol

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )

CAMERAS_FILE_PATH = Path("data/cameras.json")

DEFAULT_CAMERAS: List[CameraProfile] = [
    CameraProfile(
        camera_id="srihome_living",
        name="SriHome 2K (Living Room)",
        location_zone="Living Room",
        protocol=CameraProtocol.RTSP,
        ip_address="192.168.1.60",
        port=8554,
        username="admin",
        password="888888",
        stream_path="/profile0",
        enabled=True,
        resolution="2304x1296 QHD",
        fps=25,
    ),
    CameraProfile(
        camera_id="webcam_desk",
        name="Laptop Webcam (Escritorio / Estudio)",
        location_zone="Estudio / Escritorio",
        protocol=CameraProtocol.USB,
        usb_index=0,
        enabled=True,
        resolution="1280x720 HD",
        fps=30,
    ),
    CameraProfile(
        camera_id="ezviz_feeder",
        name="EZVIZ H8c Pro (Comedero y Bebedero)",
        location_zone="Comedero y Bebedero",
        protocol=CameraProtocol.RTSP,
        ip_address="192.168.1.55",
        port=554,
        username="admin",
        password="VERIFICATION_CODE",
        stream_path="/H.264/ch1/main",
        enabled=False,
        resolution="1920x1080 FHD",
        fps=25,
    ),
]


def load_cameras(storage_path: Path | str = CAMERAS_FILE_PATH) -> List[CameraProfile]:
    """
    Carga la lista de cámaras desde disco.
    Si el archivo no existe o está corrupto, inicializa con las cámaras por defecto.
    """
    path = Path(storage_path)
    if not path.exists():
        save_cameras(DEFAULT_CAMERAS, path)
        return list(DEFAULT_CAMERAS)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if not isinstance(raw_data, list) or len(raw_data) == 0:
                return list(DEFAULT_CAMERAS)
            cameras = [CameraProfile(**item) for item in raw_data]
            return cameras
    except Exception as err:
        logger.error("Error al cargar cámaras desde %s: %s. Reutilizando defaults.", path, err)
        return list(DEFAULT_CAMERAS)


def save_cameras(cameras: List[CameraProfile], storage_path: Path | str = CAMERAS_FILE_PATH) -> bool:
    """
    Persiste la lista de cámaras en el archivo JSON.
    """
    path = Path(storage_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            serialized = [json.loads(c.model_dump_json()) for c in cameras]
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        return True
    except Exception as err:
        logger.error("Error al guardar cámaras en %s: %s", path, err)
        return False


def get_camera_by_id(cameras: List[CameraProfile], camera_id: str) -> Optional[CameraProfile]:
    """Busca una cámara por su ID único."""
    for c in cameras:
        if c.camera_id == camera_id:
            return c
    return None


def probe_camera_connection(profile: CameraProfile, timeout_seconds: float = 4.0) -> Dict[str, Any]:
    """
    Realiza una prueba activa de conexión contra la cámara.
    Verifica handshake TCP (si es IP) y apertura de stream con OpenCV.
    
    Retorna diccionario con:
      - success (bool)
      - resolution (str | None)
      - brightness (float | None)
      - latency_ms (float | None)
      - error (str | None)
    """
    start_t = time.perf_counter()

    # 1. Si es IP (RTSP / HTTP), verificar primero conectividad TCP de socket
    if profile.protocol in (CameraProtocol.RTSP, CameraProtocol.HTTP) and profile.ip_address:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(min(2.0, timeout_seconds))
        try:
            s.connect((profile.ip_address, profile.port))
            s.close()
        except socket.timeout:
            return {
                "success": False,
                "error": f"Timeout TCP: La cámara en {profile.ip_address}:{profile.port} no respondió en 2s. Verificá que esté encendida y conectada a tu Wi-Fi.",
                "resolution": None,
                "brightness": None,
                "latency_ms": None,
            }
        except Exception as err:
            return {
                "success": False,
                "error": f"Fallo de conexión TCP a {profile.ip_address}:{profile.port}: {err}",
                "resolution": None,
                "brightness": None,
                "latency_ms": None,
            }

    # 2. Apertura de video con OpenCV
    source = profile.get_stream_source()
    cap: Optional[cv2.VideoCapture] = None
    try:
        if profile.protocol == CameraProtocol.USB:
            idx = int(profile.usb_index)
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.ole32.CoInitialize(None)
                except Exception:
                    pass
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            # RTSP / Stream
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)

        if not cap or not cap.isOpened():
            return {
                "success": False,
                "error": f"No se pudo abrir el stream de video en {profile.get_masked_url()}. Verificá credenciales y ruta del stream.",
                "resolution": None,
                "brightness": None,
                "latency_ms": None,
            }

        ret, frame = cap.read()
        elapsed = time.perf_counter() - start_t
        if not ret or frame is None:
            return {
                "success": False,
                "error": "El stream se abrió pero no transmitió fotogramas legibles. Verificá si la encriptación de video está activada o si la cámara está en standby.",
                "resolution": None,
                "brightness": None,
                "latency_ms": round(elapsed * 1000, 1),
            }

        h, w = frame.shape[:2]
        mean_b = float(frame.mean())
        res_str = f"{w}x{h}"
        if w >= 2000:
            res_str += " (2K QHD)"
        elif w >= 1900:
            res_str += " (1080p FHD)"
        elif w >= 1200:
            res_str += " (720p HD)"

        return {
            "success": True,
            "resolution": res_str,
            "brightness": round(mean_b, 1),
            "latency_ms": round(elapsed * 1000, 1),
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Excepción durante el sondeo de cámara: {exc}",
            "resolution": None,
            "brightness": None,
            "latency_ms": None,
        }
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def scan_local_lan_cameras(
    subnet_prefix: str = "192.168.1",
    ports: Tuple[int, ...] = (554, 8554),
    start_ip: int = 1,
    end_ip: int = 70,
    max_threads: int = 20,
) -> List[Dict[str, Any]]:
    """
    Escaneo rápido y ligero de la subred local buscando puertos RTSP abiertos (554, 8554).
    Útil para auto-descubrimiento de cámaras en la casa del usuario.
    """
    found: List[Dict[str, Any]] = []

    def _check_host_port(ip: str, port: int) -> Optional[Dict[str, Any]]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.35)
        try:
            res = s.connect_ex((ip, port))
            if res == 0:
                s.close()
                vendor = "SriHome / Generic" if port == 8554 else "EZVIZ / Hikvision / Tapo"
                return {"ip": ip, "port": port, "vendor_hint": vendor}
            s.close()
        except Exception:
            pass
        return None

    tasks = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for last_octet in range(start_ip, end_ip + 1):
            ip = f"{subnet_prefix}.{last_octet}"
            for port in ports:
                tasks.append(executor.submit(_check_host_port, ip, port))

        for fut in as_completed(tasks):
            res = fut.result()
            if res:
                found.append(res)

    return sorted(found, key=lambda x: (x["port"], x["ip"]))
