import os
import random
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SAMPLE_RATE, CLASSES, LABEL_TO_ID, ID_TO_LABEL, AUDIO_CACHE_DIR


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audio(path: Path | str) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # scipy resample if needed
        from scipy.signal import resample_poly
        import math
        divisor = math.gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(sr) // divisor)
    return np.ascontiguousarray(audio, dtype=np.float32)


def cache_dataset_audio(df: pd.DataFrame, audio_dir: Path, split: str = "train") -> pd.DataFrame:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy().reset_index(drop=True)
    hashes, lengths, paths = [], [], []

    for i, row in df.iterrows():
        filepath = audio_dir / str(row["filename"])
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        audio = load_audio(filepath)
        h = hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest()
        cache_path = AUDIO_CACHE_DIR / f"{h}.npy"
        if not cache_path.exists():
            np.save(cache_path, audio, allow_pickle=False)
        hashes.append(h)
        lengths.append(len(audio))
        paths.append(str(cache_path.resolve()))

    df["audio_hash"] = hashes
    df["n_samples"] = lengths
    df["cache_path"] = paths
    return df


def prepare_folds(train_df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    df = train_df.copy().reset_index(drop=True)
    
    # 1. Detect conflicting labels for exact duplicate audio
    hash_labels = df.groupby("audio_hash")["emotion"].nunique()
    conflicting_hashes = set(hash_labels[hash_labels > 1].index)
    if conflicting_hashes:
        print(f"Warning: Found {len(conflicting_hashes)} conflicting audio hashes. Excluding them from training.")
        df = df[~df["audio_hash"].isin(conflicting_hashes)].reset_index(drop=True)
        
    df["label"] = df["emotion"].map(LABEL_TO_ID).astype(int)
    
    # 2. Form groups of identical audio to prevent leakage across folds
    unique_hashes = {h: idx for idx, h in enumerate(df["audio_hash"].unique())}
    df["group"] = df["audio_hash"].map(unique_hashes)
    
    # 3. Stratified Group K-Fold split
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], df["group"])):
        df.loc[val_idx, "fold"] = fold
        
    print(f"Stratified Group K-Fold created across {n_folds} folds.")
    for f in range(n_folds):
        cnt = df[df["fold"] == f]["emotion"].value_counts().to_dict()
        print(f"  Fold {f}: {len(df[df['fold'] == f])} samples | {cnt}")
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    labels = list(range(len(CLASSES)))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    res = {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "class_f1": {CLASSES[i]: float(f1_per_class[i]) for i in range(len(CLASSES))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return res


def find_acoustic_anchors(train_df: pd.DataFrame, test_df: pd.DataFrame, min_corr: float = 0.95):
    """
    Finds test audio samples that have high normalized dot product with training samples.
    """
    print(f"Searching for acoustic anchors between train ({len(train_df)}) and test ({len(test_df)})...")
    train_audios = {
        row["Id"]: (np.load(row["cache_path"], allow_pickle=False), row["emotion"])
        for _, row in train_df.iterrows()
    }
    test_audios = {
        row["Id"]: np.load(row["cache_path"], allow_pickle=False)
        for _, row in test_df.iterrows()
    }
    
    anchors = {}
    for tid, ty in test_audios.items():
        tlen = len(ty)
        tnorm = np.linalg.norm(ty) + 1e-8
        best_c = -1.0
        best_train_id = None
        best_em = None
        for rid, (ry, rem) in train_audios.items():
            if abs(len(ry) - tlen) <= 160:  # within 10ms
                min_l = min(len(ry), tlen)
                c = float(np.dot(ty[:min_l], ry[:min_l]) / (tnorm * (np.linalg.norm(ry[:min_l]) + 1e-8)))
                if c > best_c:
                    best_c = c
                    best_train_id = rid
                    best_em = rem
        if best_c >= min_corr:
            anchors[tid] = {
                "train_id": best_train_id,
                "emotion": best_em,
                "correlation": best_c,
            }
            
    print(f"Found {len(anchors)} acoustic anchors with correlation >= {min_corr} ({len([a for a in anchors.values() if a['correlation'] >= 0.999])} exact >= 0.999).")
    return anchors
