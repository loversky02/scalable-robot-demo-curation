import sys
from pathlib import Path

# Make the repo root importable so `import robocurate` works without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
