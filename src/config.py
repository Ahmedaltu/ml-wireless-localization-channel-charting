from pathlib import Path

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# Default experiment configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# DeepMIMO placeholder configuration
DEEP_MIMO_SCENARIO = "O1_60"
MAX_SAMPLES = 5000
FEATURE_TYPE = "csi_magnitude_flattened"

# Localization defaults
WKNN_K = 5
WKNN_EPS = 1e-8

# Channel charting defaults
CC_DIM = 2
CC_N_NEIGHBORS = 15