import sys
from pathlib import Path

# Asegurar que la raiz del workspace este en sys.path para imports de paquete
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
