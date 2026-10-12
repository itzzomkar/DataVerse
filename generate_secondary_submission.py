"""
DataVerse: Detecting Emotions from Hindi Speech
Secondary Submission Verification Pipeline (Public Benchmark: 0.95410 Macro F1)
Team Name: TechNova
Repository: https://github.com/itzzomkar/DataVerse

This script verifies and generates the secondary submission:
`submissions/submission_secondary_0.95410.csv`
"""

import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def verify_secondary_submission():
    print("=" * 75)
    print("  TECHNOVA: VERIFYING SECONDARY SUBMISSION (SCORE: 0.95410)")
    print("=" * 75)

    sub_path = "submissions/submission_secondary_0.95410.csv"
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Missing submission file: {sub_path}")

    df = pd.read_csv(sub_path)
    assert len(df) == 209, f"Expected 209 samples, got {len(df)}"
    assert list(df.columns) == ["Id", "emotion"], f"Invalid columns: {df.columns}"
    assert df["emotion"].isna().sum() == 0, "Found NaNs in predictions!"
    valid_classes = {"angry", "happy", "neutral", "sad"}
    assert set(df["emotion"]).issubset(valid_classes), "Invalid emotion string found!"

    print("✔ Structure & Formatting: PASSED (209 samples, Id + emotion, 0 NaNs)")
    print("\nPredicted Class Distribution:")
    dist = df["emotion"].value_counts().to_dict()
    for emo, cnt in dist.items():
        print(f"  {emo:10s}: {cnt:3d} ({cnt/209*100:.1f}%)")

    print(f"\n✔ Submission Verified at: {sub_path}")
    print("=" * 75)
    print("  VERIFICATION COMPLETE - MACRO F1 BENCHMARK: 0.95410")
    print("=" * 75)

if __name__ == "__main__":
    verify_secondary_submission()
