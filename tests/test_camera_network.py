"""
tests/test_camera_network.py
Pruebas integrales del sistema de Red Multi-Cámara y Analítica Espacial para PawSentry AI.
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path

# Configurar path raíz
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.camera_manager import (
    DEFAULT_CAMERAS,
    get_camera_by_id,
    load_cameras,
    probe_camera_connection,
    save_cameras,
)
from core.report_generator import ReportGenerator
from core.schemas import (
    BehaviorAnalysis,
    CameraProfile,
    CameraProtocol,
    MoodType,
    PetActivityType,
)


class TestCameraNetworkSystem(unittest.TestCase):
    def setUp(self):
        self.test_cam = CameraProfile(
            camera_id="test_cam_living",
            name="Test SriHome Living",
            location_zone="Living Room",
            protocol=CameraProtocol.RTSP,
            ip_address="192.168.1.60",
            port=8554,
            username="admin",
            password="888888",
            stream_path="/profile0",
            enabled=True,
        )

    def test_camera_profile_urls(self):
        """Verifica la generación de URLs y el enmascarado seguro de contraseñas."""
        url = self.test_cam.get_stream_source()
        self.assertEqual(url, "rtsp://admin:888888@192.168.1.60:8554/profile0")

        masked = self.test_cam.get_masked_url()
        self.assertIn("••••••", masked)
        self.assertNotIn("888888", masked)

        usb_cam = CameraProfile(
            camera_id="usb_test",
            name="Webcam",
            location_zone="Desk",
            protocol=CameraProtocol.USB,
            usb_index=0,
        )
        self.assertEqual(usb_cam.get_stream_source(), 0)
        self.assertIn("USB Index 0", usb_cam.get_masked_url())

    def test_camera_persistence(self):
        """Verifica guardado y carga de cámaras en data/cameras.json."""
        test_file = ROOT_DIR / "data" / "test_cameras_temp.json"
        try:
            cameras = [self.test_cam]
            saved = save_cameras(cameras, storage_path=test_file)
            self.assertTrue(saved)

            loaded = load_cameras(storage_path=test_file)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].camera_id, "test_cam_living")
            self.assertEqual(loaded[0].location_zone, "Living Room")
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_live_probe_srihome(self):
        """Verifica sondeo en vivo contra la cámara SriHome conectada a la red."""
        probe = probe_camera_connection(self.test_cam, timeout_seconds=4.5)
        print(f"\n[LIVE PROBE TEST] SriHome 192.168.1.60: {probe}")
        self.assertTrue(probe["success"], f"Fallo sondeo: {probe.get('error')}")
        self.assertIsNotNone(probe["resolution"])
        self.assertGreater(probe["brightness"], 0.0)

    def test_spatial_habit_metrics(self):
        """Verifica que ReportGenerator calcule correctamente la distribución por zonas."""
        test_events_file = ROOT_DIR / "data" / "test_events_spatial.json"
        try:
            gen = ReportGenerator(storage_file=test_events_file)
            # Agregar eventos en diferentes zonas
            ev1 = BehaviorAnalysis(
                timestamp=datetime.now(),
                pet_detected=True,
                pet_id="ichi",
                activity=PetActivityType.RESTING,
                mood=MoodType.RELAXED,
                camera_id="srihome_living",
                camera_name="SriHome 2K",
                location_zone="Living Room",
            )
            ev2 = BehaviorAnalysis(
                timestamp=datetime.now(),
                pet_detected=True,
                pet_id="ichi",
                activity=PetActivityType.DRINKING,
                mood=MoodType.RELAXED,
                camera_id="ezviz_feeder",
                camera_name="EZVIZ Comedero",
                location_zone="Comedero y Bebedero",
            )
            ev3 = BehaviorAnalysis(
                timestamp=datetime.now(),
                pet_detected=True,
                pet_id="ichi",
                activity=PetActivityType.PLAYING,
                mood=MoodType.PLAYFUL,
                camera_id="srihome_living",
                camera_name="SriHome 2K",
                location_zone="Living Room",
            )
            gen._events = [ev1, ev2, ev3]
            metrics = gen.compute_habit_metrics(pet_id="ichi")

            self.assertIn("Living Room", metrics.zone_visits_count)
            self.assertEqual(metrics.zone_visits_count["Living Room"], 2)
            self.assertIn("Comedero y Bebedero", metrics.zone_visits_count)
            self.assertEqual(metrics.zone_visits_count["Comedero y Bebedero"], 1)
            self.assertEqual(metrics.water_visits_count, 1)
        finally:
            if test_events_file.exists():
                test_events_file.unlink()


if __name__ == "__main__":
    unittest.main()
