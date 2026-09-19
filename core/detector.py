"""
core/detector.py
Módulo de Detección en el Borde (Edge) para PawSentry AI.
Gestiona la captura de video vía OpenCV (DirectShow en Windows),
la sustracción de fondo mediante MOG2 y el guardado controlado de snapshots.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# Configuración de logging estructurado conforme a PROJECT_RULES.md
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )


class PetMotionDetector:
    """
    Detector de movimiento optimizado para el borde con filtrado de contornos
    y gestión de snapshots con cooldown.
    """

    def __init__(
        self,
        camera_source: int | str = 0,
        min_area: int = 3500,
        cooldown_seconds: float = 3.0,
        snapshots_dir: Path | str = Path("data/snapshots"),
        history: int = 500,
        var_threshold: float = 25.0,
        detect_shadows: bool = False,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
        location_zone: Optional[str] = None,
    ) -> None:
        """
        Inicializa el detector de movimiento.

        Args:
            camera_source: Índice de la cámara local (ej. 0) o URL de stream RTSP (ej. 'rtsp://admin:...@192.168.1.50:554/...').
            min_area: Área mínima en píxeles de contorno para considerar movimiento real.
            cooldown_seconds: Intervalo mínimo en segundos entre guardados de snapshots.
            snapshots_dir: Carpeta de destino para los snapshots capturados.
            history: Número de fotogramas que MOG2 recuerda para modelar el fondo.
            var_threshold: Umbral de varianza de Mahalanobis para detección de fondo.
            detect_shadows: Si es True, detecta y marca sombras (más costoso en CPU).
            camera_id: Identificador de la cámara asociada.
            camera_name: Nombre amigable de la cámara.
            location_zone: Habitación o zona de la casa.
        """
        self.camera_source: int | str = camera_source
        self.camera_index: int = int(camera_source) if (isinstance(camera_source, int) or (isinstance(camera_source, str) and camera_source.isdigit())) else 0
        self.camera_id: Optional[str] = camera_id
        self.camera_name: Optional[str] = camera_name
        self.location_zone: Optional[str] = location_zone
        self.min_area: int = min_area
        self.cooldown_seconds: float = cooldown_seconds
        self.snapshots_dir: Path = Path(snapshots_dir)
        self.history: int = history
        self.var_threshold: float = var_threshold
        self.detect_shadows: bool = detect_shadows

        # Asegurar directorio de snapshots
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._cap: Optional[cv2.VideoCapture] = None
        self._bg_subtractor: Optional[cv2.BackgroundSubtractorMOG2] = None
        self._last_capture_time: float = 0.0
        # Buffer circular de pre-roll para capturas temporales (guarda ultimos 20 fotogramas)
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=20)

        # Kernel morfológico para reducción de ruido en la máscara
        self._morph_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

        logger.info(
            "PetMotionDetector inicializado [source=%s, min_area=%d, cooldown=%.1fs, out=%s]",
            str(self.camera_source),
            self.min_area,
            self.cooldown_seconds,
            self.snapshots_dir,
        )

    def start(self) -> bool:
        """
        Abre la cámara (USB vía DirectShow en Windows o stream RTSP para cámaras IP) y prepara MOG2.

        Returns:
            bool: True si la cámara se abrió exitosamente, False en caso contrario.
        """
        try:
            is_usb = isinstance(self.camera_source, int) or (isinstance(self.camera_source, str) and str(self.camera_source).isdigit())
            if is_usb:
                idx = int(self.camera_source)
                logger.info("Abriendo camara USB indice %d con backend DirectShow...", idx)
                if sys.platform == "win32":
                    try:
                        import ctypes
                        ctypes.windll.ole32.CoInitialize(None)
                    except Exception:
                        pass
                self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self._cap or not self._cap.isOpened():
                    logger.warning("Fallo DirectShow para camara %d. Probando backend estandar...", idx)
                    self._cap = cv2.VideoCapture(idx)

                # Deteccion de camara virtual negra (dummy driver) con fallback automatico a camara 0
                if self._cap and self._cap.isOpened():
                    ret_test, frame_test = self._cap.read()
                    if ret_test and frame_test is not None:
                        brightness = float(np.mean(frame_test))
                        logger.info("Brillo detectado en camara %d: %.2f", idx, brightness)
                        if brightness < 2.0 and idx != 0:
                            logger.warning(
                                "Camara %d produce fotogramas negros (brillo: %.2f). Liberando y cambiando a camara fisica 0...",
                                idx, brightness,
                            )
                            self._cap.release()
                            self._cap = None
                            time.sleep(0.15)
                            cap_fb = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                            if cap_fb.isOpened():
                                ret_fb, frame_fb = cap_fb.read()
                                if ret_fb and frame_fb is not None and float(np.mean(frame_fb)) > 5.0:
                                    logger.info("Fallback a Camara 0 exitoso (brillo: %.2f). Reemplazando fuente.", float(np.mean(frame_fb)))
                                    self._cap = cap_fb
                                    self.camera_source = 0
                                    self.camera_index = 0
                                else:
                                    cap_fb.release()
            else:
                src_url = str(self.camera_source)
                logger.info("Abriendo stream de red/RTSP (IP Cam): %s...", src_url)
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;2000000"
                self._cap = cv2.VideoCapture(src_url, cv2.CAP_FFMPEG)

            if not self._cap or not self._cap.isOpened():
                logger.error("No se pudo abrir la fuente de video: %s", str(self.camera_source))
                self._cap = None
                return False

            # Configuración recomendada de resolución estándar (640x480 para latencia mínima)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )

            logger.info("Fuente de video %s y MOG2 inicializados exitosamente.", str(self.camera_source))
            return True

        except Exception as err:
            logger.error("Error al inicializar la camara: %s", err, exc_info=True)
            self.stop()
            return False

    def can_capture(self) -> bool:
        """Indica si ha transcurrido el tiempo de cooldown para permitir una nueva captura."""
        return (time.time() - self._last_capture_time) >= self.cooldown_seconds

    def get_cooldown_remaining(self) -> float:
        """Retorna los segundos restantes de cooldown (0.0 si ya esta disponible)."""
        elapsed = time.time() - self._last_capture_time
        return max(0.0, self.cooldown_seconds - elapsed)

    def is_active(self) -> bool:
        """Indica si la cámara está activa y disponible."""
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee el siguiente fotograma del stream y lo anade al buffer circular.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (exito, fotograma_bgr)
        """
        if not self.is_active():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Fallo al leer fotograma de la camara %d", self.camera_index)
            return False, None

        self._frame_buffer.append(frame.copy())
        return True, frame

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Aplica sustracción de fondo, limpieza morfológica y extracción de cajas de movimiento.

        Args:
            frame: Fotograma original en formato BGR.

        Returns:
            Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]]]:
                - motion_detected: True si hay contornos que superen min_area.
                - mask: Máscara binaria del movimiento.
                - bounding_boxes: Lista de tuplas (x, y, w, h) de las regiones con movimiento.
        """
        if self._bg_subtractor is None:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )

        # 1. Desenfoque suave para eliminar ruido de alta frecuencia
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)

        # 2. Aplicar MOG2
        fg_mask = self._bg_subtractor.apply(blurred)

        # 3. Operaciones morfológicas: apertura para eliminar motas, dilatación para fusionar masas
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
        fg_mask = cv2.dilate(fg_mask, self._morph_kernel, iterations=2)

        # 4. Encontrar contornos de movimiento
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bounding_boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))

        motion_detected = len(bounding_boxes) > 0
        return motion_detected, fg_mask, bounding_boxes

    def capture_snapshot(
        self, frame: np.ndarray, prefix: str = "pet"
    ) -> Optional[Path]:
        """
        Guarda un snapshot en disco si ha transcurrido el tiempo de cooldown.

        Args:
            frame: Fotograma que se desea persistir.
            prefix: Prefijo del nombre del archivo.

        Returns:
            Optional[Path]: Ruta del archivo generado, o None si el cooldown está activo.
        """
        now = time.time()
        elapsed = now - self._last_capture_time

        if elapsed < self.cooldown_seconds:
            logger.debug(
                "Captura omitida por cooldown (transcurridos: %.2fs / req: %.2fs)",
                elapsed,
                self.cooldown_seconds,
            )
            return None

        # Generar nombre único con timestamp microsegundo
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = self.snapshots_dir / f"{prefix}_{timestamp_str}.jpg"

        try:
            success = cv2.imwrite(str(file_path), frame)
            if success:
                self._last_capture_time = now
                logger.info("Snapshot guardado con exito: %s", file_path)
                return file_path
            else:
                logger.error("cv2.imwrite fallo al escribir en: %s", file_path)
                return None
        except Exception as err:
            logger.error("Excepcion al guardar snapshot: %s", err, exc_info=True)
            return None

    def capture_micro_event_burst(
        self, current_frame: np.ndarray, prefix: str = "burst"
    ) -> List[Path]:
        """
        Captura una rafaga temporal de 3 fotogramas clave (inicio del movimiento,
        fotograma actual y final) para que los modelos multimodales evalúen dinámica y trayectoria.

        Args:
            current_frame: Fotograma actual en el momento de activación.
            prefix: Prefijo para los nombres de archivo.

        Returns:
            List[Path]: Rutas de los fotogramas capturados (onset, peak, post).
        """
        now = time.time()
        elapsed = now - self._last_capture_time

        if elapsed < self.cooldown_seconds:
            return []

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        burst_paths: List[Path] = []

        # 1. Fotograma inicial (pre-roll extraído del buffer temporal)
        onset_frame = self._frame_buffer[0] if len(self._frame_buffer) > 0 else current_frame
        path_onset = self.snapshots_dir / f"{prefix}_{timestamp_str}_01_onset.jpg"
        if cv2.imwrite(str(path_onset), onset_frame):
            burst_paths.append(path_onset)

        # 2. Fotograma actual (punto álgido del movimiento)
        path_peak = self.snapshots_dir / f"{prefix}_{timestamp_str}_02_peak.jpg"
        if cv2.imwrite(str(path_peak), current_frame):
            burst_paths.append(path_peak)

        # 3. Fotograma posterior (post-roll capturado tras breve intervalo de 150ms)
        time.sleep(0.15)
        ret, post_frame = self.read_frame()
        final_frame = post_frame if (ret and post_frame is not None) else current_frame
        path_post = self.snapshots_dir / f"{prefix}_{timestamp_str}_03_post.jpg"
        if cv2.imwrite(str(path_post), final_frame):
            burst_paths.append(path_post)

        if burst_paths:
            self._last_capture_time = time.time()
            logger.info("Rafaga de micro-evento capturada (%d fotogramas): %s", len(burst_paths), burst_paths)

        return burst_paths

    def switch_camera(
        self,
        new_source: int | str,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
        location_zone: Optional[str] = None,
    ) -> bool:
        """
        Cambia dinámicamente la fuente de video en caliente,
        reiniciando el capturador y el sustractor de fondo MOG2.
        """
        self.stop()
        self.camera_source = new_source
        self.camera_index = int(new_source) if (isinstance(new_source, int) or (isinstance(new_source, str) and str(new_source).isdigit())) else 0
        if camera_id is not None:
            self.camera_id = camera_id
        if camera_name is not None:
            self.camera_name = camera_name
        if location_zone is not None:
            self.location_zone = location_zone
        self._frame_buffer.clear()
        logger.info(
            "Cámara conmutada a: %s (ID: %s, Zona: %s)",
            str(self.camera_source),
            str(self.camera_id),
            str(self.location_zone),
        )
        return self.start()

    def stop(self) -> None:
        """Libera la cámara y los recursos de memoria."""
        if self._cap is not None:
            logger.info("Liberando camara %d...", self.camera_index)
            try:
                self._cap.release()
            except Exception as err:
                logger.warning("Error al liberar camara: %s", err)
            finally:
                self._cap = None

        self._bg_subtractor = None
        logger.info("Detector detenido.")

    def __enter__(self) -> PetMotionDetector:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()


if __name__ == "__main__":
    import sys

    print("==================================================")
    print("      PawSentry AI — Test de Edge Detector        ")
    print("==================================================")
    print("Presiona 'q' en la ventana de video para salir.")

    detector = PetMotionDetector(
        camera_index=0,
        min_area=3000,
        cooldown_seconds=3.0,
        snapshots_dir=Path("data/snapshots"),
    )

    if not detector.start():
        print("[ERROR] No se pudo inicializar la camara.")
        sys.exit(1)

    try:
        frame_count = 0
        while True:
            ret, frame = detector.read_frame()
            if not ret or frame is None:
                print("[!] No se pudo obtener fotograma.")
                break

            frame_count += 1
            motion_detected, mask, bboxes = detector.process_frame(frame)

            # Dibujar rectángulos y HUD sobre el frame
            display_frame = frame.copy()
            status_text = "ESTADO: EN ESPERA"
            hud_color = (0, 255, 0)

            if motion_detected:
                status_text = f"ESTADO: MOVIMIENTO DETECTADO ({len(bboxes)} regiones)"
                hud_color = (0, 0, 255)
                for (x, y, w, h) in bboxes:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"Area: {w*h}",
                        (x, max(20, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                # Intentar captura automática con cooldown
                snapshot_path = detector.capture_snapshot(frame, prefix="pet_event")
                if snapshot_path:
                    print(f"[*] Snapshot capturado: {snapshot_path}")

            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hud_color,
                2,
            )

            cv2.imshow("PawSentry AI - Live Edge Monitor", display_frame)
            cv2.imshow("PawSentry AI - Motion Mask (MOG2)", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[*] Salida solicitada por el usuario.")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupcion por teclado.")
    finally:
        detector.stop()
        cv2.destroyAllWindows()
        print("[*] Detector finalizado limpiamente.")
