"""Configuration system for Business Entity Resolution pipeline.

Supports seamless switching between:
- Local smoke test mode (small configurable sample, fast, 8 GB RAM friendly)
- Full-scale execution mode (designed for Google Colab / server environments)

All parameters can be configured via environment variables or CLI arguments.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


def get_bool_env(var_name: str, default: bool = False) -> bool:
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "t", "y")


@dataclass
class PipelineConfig:
    # Directory paths
    DATA_ROOT: str = os.getenv("DATA_ROOT", "dataset")
    TRAIN_DIR: str = os.getenv("TRAIN_DIR", "")
    TEST_DIR: str = os.getenv("TEST_DIR", "")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "cache")

    # Execution modes
    LOCAL_SMOKE_TEST: bool = get_bool_env("LOCAL_SMOKE_TEST", False)
    SMOKE_SAMPLE_SIZE: int = int(os.getenv("SMOKE_SAMPLE_SIZE", "1000"))
    
    # Random seed
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    
    # Hardware / Multiprocessing
    N_JOBS: int = int(os.getenv("N_JOBS", "-1"))
    
    # Blocking parameters
    BLOCKING_MAX_CANDIDATES: int = int(os.getenv("BLOCKING_MAX_CANDIDATES", "60"))
    NAME_TFIDF_MIN_SIM: float = float(os.getenv("NAME_TFIDF_MIN_SIM", "0.45"))
    MIN_TOKEN_LEN: int = 3
    
    # Model & Calibration parameters
    CALIBRATED_THRESHOLD: float = float(os.getenv("CALIBRATED_THRESHOLD", "0.50"))
    VALIDATION_SPLIT_RATIO: float = float(os.getenv("VALIDATION_SPLIT_RATIO", "0.20"))

    def __post_init__(self):
        # Resolve train and test dirs relative to DATA_ROOT if not explicitly set
        if not self.TRAIN_DIR:
            self.TRAIN_DIR = os.path.join(self.DATA_ROOT, "train")
        if not self.TEST_DIR:
            self.TEST_DIR = os.path.join(self.DATA_ROOT, "test")
            
        # Ensure directories exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)


def get_default_config() -> PipelineConfig:
    return PipelineConfig()
