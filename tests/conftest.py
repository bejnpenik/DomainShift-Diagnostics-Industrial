import sys
from pathlib import Path

# Add the project root (parent of tests/) to sys.path
# so tests can import collection, metrics, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
