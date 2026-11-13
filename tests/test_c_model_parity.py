#!/usr/bin/env python3
"""
Test C Model Parity against Python Scikit-Learn Random Forest
Monsoon Sentinel - Step 2 Verification Test

Verifies that the static embedded C code generated in Core/Src/model_inference.c
achieves 100.0% parity against the Python reference model across 100 randomly
sampled physical feature windows.
"""

from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "master_features_v4.csv"
INC_DIR = PROJECT_ROOT / "Core" / "Inc"
SRC_FILE = PROJECT_ROOT / "Core" / "Src" / "model_inference.c"

FEATURE_COLS = [
    "tilt_deg",
    "pitch_deg",
    "roll_deg",
    "var_dyn_g2",
    "rms_dyn_g",
    "p2p_dyn_g",
    "crest_factor",
    "gyro_rms_dps",
    "dominant_freq_hz",
    "band_low",
    "band_mid",
    "band_high",
    "spectral_ratio_low",
]


@pytest.fixture(scope="module")
def compiled_c_binary(tmp_path_factory):
    """Compiles a standalone native test harness with model_inference.c."""
    build_dir = tmp_path_factory.mktemp("c_build")
    harness_src = build_dir / "harness.c"
    binary_path = build_dir / "c_model_runner"

    # Minimal C harness reading 13 floats per line from stdin and emitting predicted class
    harness_code = """
#include <stdio.h>
#include <stdlib.h>
#include "model_inference.h"

int main(void) {
    float f[13];
    while (scanf("%f %f %f %f %f %f %f %f %f %f %f %f %f",
                 &f[0], &f[1], &f[2], &f[3], &f[4], &f[5],
                 &f[6], &f[7], &f[8], &f[9], &f[10], &f[11], &f[12]) == 13) {
        uint8_t pred = model_predict(f);
        printf("%u\\n", (unsigned int)pred);
    }
    return 0;
}
"""
    harness_src.write_text(harness_code, encoding="utf-8")

    compile_cmd = [
        "gcc",
        "-O2",
        f"-I{INC_DIR}",
        str(SRC_FILE),
        str(harness_src),
        "-o",
        str(binary_path),
    ]

    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"Failed to compile C harness: {res.stderr}")

    return binary_path


def test_c_vs_python_parity(compiled_c_binary):
    """
    Feeds 100 randomly sampled windows to both Python and C,
    asserting 100% bitwise class equivalence.
    """
    assert CSV_PATH.exists(), f"Processed CSV not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)

    # Filter Regime 2 and remap labels to 4-class kinematics
    df = df[df["label"] != 2].copy()
    label_remap = {0: 0, 1: 1, 3: 2, 4: 3}
    df["label"] = df["label"].map(label_remap)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # Fit production Random Forest reference model
    rf = RandomForestClassifier(
        n_estimators=40,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X, y)

    # Sample 100 feature vectors deterministically
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(X), 100, replace=False)
    X_sample = X[sample_indices]

    # 1. Compute Python predictions
    py_preds = rf.predict(X_sample).astype(int)

    # 2. Pipe to C runner
    input_str = "\n".join(" ".join(f"{val:.7f}" for val in row) for row in X_sample)
    proc = subprocess.run(
        [str(compiled_c_binary)],
        input=input_str,
        capture_output=True,
        text=True,
        check=True,
    )

    c_preds = [int(line.strip()) for line in proc.stdout.strip().splitlines() if line.strip()]

    # Verification assertions
    assert len(c_preds) == 100, f"Expected 100 outputs from C runner, got {len(c_preds)}"
    matches = (np.array(c_preds) == py_preds)
    match_pct = np.mean(matches) * 100.0

    print(f"\n[Parity Check] Matches: {np.sum(matches)} / 100 ({match_pct:.2f}%)")
    assert np.array_equal(c_preds, py_preds), (
        f"Mismatch detected between C and Python: {match_pct:.2f}% match"
    )
