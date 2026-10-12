import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "SRCASW-BhavVaani"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_AUDIO_DIR = DATA_DIR / "train"
TEST_AUDIO_DIR = DATA_DIR / "test"

EXPERIMENTS_DIR = BASE_DIR / "experiments"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"

# Audio parameters
SAMPLE_RATE = 16000
MAX_SECONDS = 8.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)
MIN_SAMPLES = 4000

# Classes
CLASSES = ["angry", "happy", "neutral", "sad"]
NUM_CLASSES = len(CLASSES)
LABEL_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}
ID_TO_LABEL = {idx: cls for idx, cls in enumerate(CLASSES)}

# Training hyperparameters
MODEL_NAME = "microsoft/wavlm-base-plus"
N_FOLDS = 5
SEED = 42

BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EPOCHS = 18
MIN_EPOCHS = 6
HEAD_WARMUP_EPOCHS = 1

BACKBONE_LR = 2e-5
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05

EVAL_CROPS = [0.0, 0.5, 1.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
