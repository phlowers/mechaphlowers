import sys
from pathlib import Path

projet_dir: Path = Path(__file__).resolve().parents[1]
source_dir: Path = projet_dir / "src"
sys.path.insert(0, str(source_dir))
