import os
import json
import gc
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from transformers import AutoFeatureExtractor, get_cosine_schedule_with_warmup

from src.config import (
    TRAIN_CSV,
    TRAIN_AUDIO_DIR,
    EXPERIMENTS_DIR,
    MODEL_NAME,
    N_FOLDS,
    SEED,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    ACCUMULATION_STEPS,
    EPOCHS,
    MIN_EPOCHS,
    HEAD_WARMUP_EPOCHS,
    BACKBONE_LR,
    HEAD_LR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    EVAL_CROPS,
    DEVICE,
    USE_AMP,
    CLASSES,
)
from src.utils import seed_everything, cache_dataset_audio, prepare_folds, compute_metrics
from src.dataset import get_dataloader
from src.models import SpeechEmotionModel


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    accumulation_steps: int = ACCUMULATION_STEPS,
    label_smoothing: float = LABEL_SMOOTHING,
    freeze_backbone: bool = False,
) -> float:
    model.train()
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        for p in model.backbone.parameters():
            p.requires_grad = True
            
    total_loss = 0.0
    num_samples = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_values = batch["input_values"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(input_values, attention_mask)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            loss_scaled = loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * len(labels)
        num_samples += len(labels)

    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader, num_samples: int) -> np.ndarray:
    model.eval()
    num_classes = len(CLASSES)
    sums = np.zeros((num_samples, num_classes), dtype=np.float64)
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

    probabilities = (sums / counts[:, None]).astype(np.float32)
    return probabilities


def run_training(
    model_name: str = MODEL_NAME,
    exp_name: str = "wavlm_baseline",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    folds_to_run: list = None,
):
    seed_everything(seed)
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Starting Training Experiment: {exp_name} ===")
    print(f"Backbone: {model_name} | Device: {DEVICE} | AMP: {USE_AMP}")

    # Load and cache audio
    raw_train = pd.read_csv(TRAIN_CSV)
    df = cache_dataset_audio(raw_train, TRAIN_AUDIO_DIR, split="train")
    df = prepare_folds(df, n_folds=N_FOLDS, seed=seed)
    df.to_csv(exp_dir / "train_with_folds.csv", index=False)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extractor.save_pretrained(exp_dir / "feature_extractor")

    oof_probs = np.full((len(df), len(CLASSES)), np.nan, dtype=np.float32)
    fold_metrics = {}
    
    if folds_to_run is None:
        folds_to_run = list(range(N_FOLDS))

    for fold in folds_to_run:
        print(f"\n===================== Fold {fold} / {N_FOLDS} =====================")
        train_sub = df[df["fold"] != fold].reset_index(drop=True)
        val_sub = df[df["fold"] == fold].reset_index(drop=True)
        val_indices = np.where(df["fold"] == fold)[0]

        train_loader = get_dataloader(
            train_sub, feature_extractor, batch_size=batch_size,
            training=True, augment=True, seed=seed + fold
        )
        val_loader = get_dataloader(
            val_sub, feature_extractor, batch_size=EVAL_BATCH_SIZE,
            training=False, crops=EVAL_CROPS, augment=False
        )

        model = SpeechEmotionModel(model_name=model_name).to(DEVICE)

        # Differential learning rates
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ])

        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler("cuda", enabled=USE_AMP)

        best_macro_f1 = -1.0
        best_epoch = -1
        best_probs = None
        checkpoint_path = exp_dir / f"best_model_fold_{fold}.pt"

        for epoch in range(1, epochs + 1):
            freeze_backbone = (epoch <= HEAD_WARMUP_EPOCHS)
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                freeze_backbone=freeze_backbone
            )

            val_probs = evaluate(model, val_loader, len(val_sub))
            val_preds = val_probs.argmax(axis=1)
            metrics = compute_metrics(val_sub["label"].to_numpy(), val_preds, val_probs)

            f1 = metrics["macro_f1"]
            acc = metrics["accuracy"]
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro F1: {f1:.4f} | Val Acc: {acc:.4f} | Per-class: {metrics['class_f1']}")

            if f1 > best_macro_f1:
                best_macro_f1 = f1
                best_epoch = epoch
                best_probs = val_probs
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "macro_f1": best_macro_f1,
                    "metrics": metrics,
                    "model_name": model_name,
                }, checkpoint_path)

        print(f"--> Fold {fold} Complete! Best Macro F1: {best_macro_f1:.4f} at Epoch {best_epoch}")
        oof_probs[val_indices] = best_probs
        fold_metrics[f"fold_{fold}"] = {
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
        }

        del model, optimizer, scheduler, scaler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Overall Out-of-Fold Evaluation
    valid_mask = ~np.isnan(oof_probs[:, 0])
    y_true = df.loc[valid_mask, "label"].to_numpy()
    y_pred = oof_probs[valid_mask].argmax(axis=1)
    overall_metrics = compute_metrics(y_true, y_pred, oof_probs[valid_mask])

    print("\n===================== Overall OOF Performance =====================")
    print(f"Overall OOF Macro F1: {overall_metrics['macro_f1']:.4f}")
    print(f"Overall OOF Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Per-Class F1: {overall_metrics['class_f1']}")
    print(f"Confusion Matrix:\n{np.array(overall_metrics['confusion_matrix'])}")

    # Save OOF predictions
    oof_df = df.copy()
    for i, cls in enumerate(CLASSES):
        oof_df[f"prob_{cls}"] = oof_probs[:, i]
    oof_df["pred_emotion"] = ""
    oof_df.loc[valid_mask, "pred_emotion"] = [CLASSES[p] for p in y_pred]
    oof_df.to_csv(exp_dir / "oof_predictions.csv", index=False)

    summary = {
        "model_name": model_name,
        "seed": seed,
        "overall_macro_f1": overall_metrics["macro_f1"],
        "overall_accuracy": overall_metrics["accuracy"],
        "per_class_f1": overall_metrics["class_f1"],
        "fold_metrics": fold_metrics,
    }
    with open(exp_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return overall_metrics["macro_f1"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--exp-name", default="wavlm_base_plus_run1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    run_training(
        model_name=args.model_name,
        exp_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        folds_to_run=args.folds,
    )
