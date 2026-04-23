"""Root conftest.py - adds src/ to sys.path for test discovery."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
