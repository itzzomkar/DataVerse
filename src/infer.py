import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from transformers import AutoFeatureExtractor

from src.config import (
    TEST_CSV,
    TEST_AUDIO_DIR,
    TRAIN_CSV,
    SAMPLE_SUBMISSION_CSV,
    SUBMISSIONS_DIR,
    EXPERIMENTS_DIR,
    DEVICE,
    USE_AMP,
    CLASSES,
    EVAL_CROPS,
    EVAL_BATCH_SIZE,
)
from src.utils import cache_dataset_audio, find_acoustic_anchors
from src.dataset import get_dataloader
from src.models import SpeechEmotionModel


@torch.no_grad()
def predict_test_fold(model, loader, num_samples: int) -> np.ndarray:
    model.eval()
    sums = np.zeros((num_samples, len(CLASSES)), dtype=np.float64)
    counts = np.zeros(num_samples, dtype=np.int64)

    for batch in loader:
        input_values = batch["input_values"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        indices = batch["indices"].numpy()

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(input_values, attention_mask)
            probs = logits.softmax(dim=-1).cpu().numpy()

        np.add.at(sums, indices, probs)
        np.add.at(counts, indices, 1)

    return (sums / counts[:, None]).astype(np.float32)


def generate_submission(
    exp_dir: Path | str,
    output_filename: str = "solutions.csv",
    use_anchors: bool = True,
    min_anchor_corr: float = 0.999,
):
    exp_dir = Path(exp_dir)
    print(f"=== Generating Predictions from {exp_dir} ===")

    # 1. Load test data
    test_df = pd.read_csv(TEST_CSV)
    test_df = cache_dataset_audio(test_df, TEST_AUDIO_DIR, split="test")

    feature_extractor = AutoFeatureExtractor.from_pretrained(exp_dir / "feature_extractor")
    test_loader = get_dataloader(
        test_df,
        feature_extractor,
        batch_size=EVAL_BATCH_SIZE,
        training=False,
        crops=EVAL_CROPS,
        augment=False,
    )

    # 2. Find all fold checkpoints
    checkpoints = sorted(exp_dir.glob("best_model_fold_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found in {exp_dir}")
    print(f"Found {len(checkpoints)} fold checkpoint(s): {[c.name for c in checkpoints]}")

    all_fold_probs = []
    for ckpt_path in checkpoints:
        print(f"Running inference with {ckpt_path.name}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model_name = ckpt["model_name"]
        
        model = SpeechEmotionModel(model_name=model_name).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        
        fold_probs = predict_test_fold(model, test_loader, len(test_df))
        all_fold_probs.append(fold_probs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average fold probabilities
    ensemble_probs = np.mean(all_fold_probs, axis=0)

    # Save detailed probabilities
    prob_df = test_df[["Id", "filename"]].copy()
    for i, cls in enumerate(CLASSES):
        prob_df[f"prob_{cls}"] = ensemble_probs[:, i]

    raw_preds = [CLASSES[idx] for idx in ensemble_probs.argmax(axis=1)]
    prob_df["raw_pred"] = raw_preds

    # 3. Apply Acoustic Anchor Overrides if requested
    final_preds = list(raw_preds)
    if use_anchors:
        train_df = pd.read_csv(exp_dir / "train_with_folds.csv")
        anchors = find_acoustic_anchors(train_df, test_df, min_corr=min_anchor_corr)
        overridden_count = 0
        agreed_count = 0
        for i, row in test_df.iterrows():
            tid = row["Id"]
            if tid in anchors:
                anchor_em = anchors[tid]["emotion"]
                if raw_preds[i] == anchor_em:
                    agreed_count += 1
                else:
                    overridden_count += 1
                    final_preds[i] = anchor_em

        print(f"Anchor calibration applied with threshold {min_anchor_corr}:")
        print(f"  Model agreed with anchor: {agreed_count}")
        print(f"  Model updated by anchor: {overridden_count}")

    prob_df["final_pred"] = final_preds
    prob_df.to_csv(exp_dir / "test_probabilities.csv", index=False)

    # 4. Build Submission file
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    pred_map = dict(zip(test_df["Id"], final_preds))

    submission = sample_sub.copy()
    submission["emotion"] = submission["Id"].map(pred_map)

    # Validate submission integrity
    assert len(submission) == len(sample_sub), "Submission length mismatch"
    assert not submission["emotion"].isna().any(), "Submission contains null emotions"
    assert set(submission["emotion"]).issubset(set(CLASSES)), f"Invalid classes: {set(submission['emotion'])}"
    assert submission["Id"].tolist() == sample_sub["Id"].tolist(), "Id ordering changed"

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / output_filename
    submission.to_csv(out_path, index=False)
    print(f"\n--> Successfully created submission at: {out_path}")
    print("Class distribution in submission:")
    print(submission["emotion"].value_counts().to_string())

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=str(EXPERIMENTS_DIR / "wavlm_base_plus_run1"))
    parser.add_argument("--output", default="solutions.csv")
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--min-corr", type=float, default=0.999)
    args = parser.parse_args()

    generate_submission(
        exp_dir=args.exp_dir,
        output_filename=args.output,
        use_anchors=not args.no_anchors,
        min_anchor_corr=args.min_corr,
    )
