import cv2
import sys

def test_cameras(max_tested=3):
    found_cameras = []
    print("=== Escaneando camaras disponibles ===")
    for index in range(max_tested):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW es mas rapido y confiable en Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, _ = frame.shape
                filename = f"test_camera_{index}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] Camara detectada en indice {index}: Resolucion {w}x{h}. Fotograma guardado en {filename}")
                found_cameras.append(index)
            else:
                print(f"[!] Camara en indice {index} abierta pero no pudo leer fotograma.")
            cap.release()
        else:
            print(f"[-] No hay camara en indice {index}.")
            
    if found_cameras:
        print(f"\nExito! Se encontraron {len(found_cameras)} camara(s): {found_cameras}")
    else:
        print("\n[Aviso] No se detecto ninguna camara en los indices 0-{max_tested-1}.")

if __name__ == '__main__':
    test_cameras()
