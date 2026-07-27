"""Application configuration and project paths."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PACKAGE_ROOT / "data" / "processed_data.csv"
MODEL_PATH = PACKAGE_ROOT / "models" / "neural_ranker.pt"
LOG_DIR = PACKAGE_ROOT / "logs"
LOG_PATH = LOG_DIR / "api.log"

DEFAULT_CANDIDATE_POOL_SIZE = 100
DEFAULT_TOP_K = 10
