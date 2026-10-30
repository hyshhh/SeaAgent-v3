import sys
from pathlib import Path

ETA_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ETA_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)
