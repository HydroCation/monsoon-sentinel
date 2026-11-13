#!/usr/bin/env python3
"""
Monsoon Sentinel Edge Classification & Benchmark Suite
Part 2: Edge DSP, Physical Feature Engineering, & Regime Classification

Cross-Validation Enhancement:
  - Temporal Session Block Grouping:
    * Files with multiple recordings retain `source_file` as group key.
    * `mpu6050_flat_taps_1.csv` (Impulse_Shock, 189 windows) is partitioned into 4
      contiguous temporal blocks (approx 47-48 windows each) to prevent group
      starvation in StratifiedGroupKFold while maintaining temporal boundary isolation.
  - Dual Regime Evaluation:
    * Option A (4-Class): Nominal_Baseline, Structural_Tilt, Impulse_Shock, Mast_Buffeting
    * Option B (3-Class): Nominal_Baseline, Structural_Tilt, Mast_Buffeting (Shock excluded)

Tuned Models Evaluated:
  - Linear SVM: StandardScaler + LinearSVC(C=1.0, class_weight='balanced')
  - Random Forest: RandomForestClassifier(n_estimators=40, max_depth=7, class_weight='balanced')
  - Edge MLP: StandardScaler + MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500)

Validation Strategy:
  - 5-Fold StratifiedGroupKFold on `group` (source files + temporal shock blocks)
  - Strict Out-Of-Fold (OOF) prediction aggregation

Outputs:
  - Dual benchmark tables (stdout & LaTeX)
  - High-resolution 8-panel comparison figure: reports/figures/model_benchmark_confusion_matrices.png (300 DPI)
  - Updated LaTeX benchmark table: reports/part2_classifier/table_model_benchmarks.tex
  - Best model parameter/weight export: data/processed/best_model_weights.json
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ==============================================================================
# 0. DIRECTORY & DATA SETUP
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "master_features_v4.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PART2_DIR = REPORTS_DIR / "part2_classifier"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PART2_DIR.mkdir(parents=True, exist_ok=True)

CM_PLOT_PATH = FIGURES_DIR / "model_benchmark_confusion_matrices.png"
WEIGHTS_JSON_PATH = PROCESSED_DIR / "best_model_weights.json"
WEIGHTS_OPT_B_PATH = PROCESSED_DIR / "best_model_weights_option_b.json"
LATEX_TABLE_PATH = PART2_DIR / "table_model_benchmarks.tex"

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

CLASS_NAMES_A = [
    "Nominal_Baseline",
    "Structural_Tilt",
    "Impulse_Shock",
    "Mast_Buffeting",
]
DISPLAY_NAMES_A = [
    "Baseline",
    "Tilt",
    "Shock",
    "Buffeting",
]

CLASS_NAMES_B = [
    "Nominal_Baseline",
    "Structural_Tilt",
    "Mast_Buffeting",
]
DISPLAY_NAMES_B = [
    "Baseline",
    "Tilt",
    "Buffeting",
]


def prepare_datasets():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Processed feature dataset not found: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)
    print("=" * 85)
    print(f"LOADING DATASET: {INPUT_CSV.name} ({len(df_raw)} initial rows)")
    print("=" * 85)

    # 1. Exclude Regime 2 (Acoustic_Vibration / original label 2)
    df = df_raw[df_raw["label"] != 2].copy()

    # 2. Temporal Session Block Grouping
    # For mpu6050_flat_taps_1.csv (189 rows), split contiguous temporal stream into 4 blocks
    df["group"] = df["source_file"].copy()
    taps_mask = df["source_file"] == "mpu6050_flat_taps_1.csv"
    taps_indices = df[taps_mask].index
    splits = np.array_split(taps_indices, 4)
    for i, block_indices in enumerate(splits, 1):
        df.loc[block_indices, "group"] = f"flat_taps_block_{i}"

    print(f"Filtered dataset (excluding Acoustic_Vibration): {len(df)} rows")
    print(f"Total distinct cross-validation groups: {df['group'].nunique()}")
    print("  - Standard file groups: 19")
    print(f"  - Shock session temporal blocks: 4 ({[len(b) for b in splits]} samples each)")

    # Option A: 4-Class Kinematics
    # 0 -> 0 (Baseline), 1 -> 1 (Tilt), 3 -> 2 (Shock), 4 -> 3 (Buffeting)
    df_a = df.copy()
    label_remap_a = {0: 0, 1: 1, 3: 2, 4: 3}
    df_a["label"] = df_a["label"].map(label_remap_a)

    # Option B: 3-Class Kinematics (excluding Shock entirely)
    # 0 -> 0 (Baseline), 1 -> 1 (Tilt), 4 -> 2 (Buffeting)
    df_b = df[df["label"] != 3].copy()
    label_remap_b = {0: 0, 1: 1, 4: 2}
    df_b["label"] = df_b["label"].map(label_remap_b)

    return df_a, df_b


def run_cv_benchmark(df: pd.DataFrame, option_name: str, class_names: list, display_names: list):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["group"].values
    n_samples = len(y)
    n_classes = len(class_names)

    print("\n" + "=" * 85)
    print(f"EXECUTING 5-FOLD STRATIFIED GROUP CV: {option_name}")
    print(f"Dataset: {n_samples} samples across {df['group'].nunique()} groups ({n_classes} classes)")
    print("=" * 85)
    for cid, cname in enumerate(class_names):
        count = (y == cid).sum()
        pct = count / n_samples * 100.0
        n_grps = df[df["label"] == cid]["group"].nunique()
        print(f"  Class {cid} ({cname:<18}): {count:>5} samples ({pct:>5.2f}%) across {n_grps} groups")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    model_factories = {
        "Linear SVM": lambda: LinearSVC(
            dual=False, C=1.0, class_weight="balanced", random_state=42, max_iter=2000
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=40,
            max_depth=7,
            class_weight="balanced",
            random_state=42,
        ),
        "Edge MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=500, random_state=42
        ),
    }

    results = {}
    oof_predictions = {}
    fitted_models = {}

    for name, factory in model_factories.items():
        print(f"\n[{option_name}] Training {name} across 5 folds...")
        oof_preds = np.zeros(n_samples, dtype=int)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            y_train = y[train_idx]
            y_val = y[val_idx]

            clf = factory()
            if "Random Forest" in name:
                clf.fit(X[train_idx], y_train)
                preds = clf.predict(X[val_idx])
            else:
                clf.fit(X_train, y_train)
                preds = clf.predict(X_val)

            oof_preds[val_idx] = preds
            f_score = f1_score(y_val, preds, average="macro", zero_division=0)
            fold_scores.append(f_score)
            print(f"  Fold {fold+1}/5: Macro-F1 = {f_score:.4f} (val: {len(val_idx)} samples)")

        b_acc = balanced_accuracy_score(y, oof_preds)
        prec = precision_score(y, oof_preds, average="macro", zero_division=0)
        rec = recall_score(y, oof_preds, average="macro", zero_division=0)
        f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
        cm = confusion_matrix(y, oof_preds, labels=list(range(n_classes)))
        report_str = classification_report(y, oof_preds, target_names=display_names, digits=4)

        results[name] = {
            "balanced_accuracy": b_acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "confusion_matrix": cm,
            "fold_scores": fold_scores,
            "classification_report": report_str,
        }
        oof_predictions[name] = oof_preds

        # Fit model on entire option dataset for production C export
        full_scaler = StandardScaler()
        full_X = full_scaler.fit_transform(X)
        full_clf = factory()
        if "Random Forest" in name:
            full_clf.fit(X, y)
        else:
            full_clf.fit(full_X, y)
        fitted_models[name] = (full_clf, full_scaler)

    return results, oof_predictions, fitted_models, X, y


def print_comparison_tables(results_a: dict, results_b: dict):
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON TABLE: OPTION A (4-CLASS WITH TEMPORAL SHOCK GROUPING)")
    print("=" * 90)
    header = f"{'Model Architecture':<20} | {'Balanced Acc':<14} | {'Macro Prec':<14} | {'Macro Rec':<14} | {'Macro-F1':<10}"
    print(header)
    print("-" * 90)
    for name, m in results_a.items():
        print(f"{name:<20} | {m['balanced_accuracy']*100:>12.2f}% | {m['macro_precision']*100:>12.2f}% | {m['macro_recall']*100:>12.2f}% | {m['macro_f1']*100:>8.2f}%")
    print("=" * 90)

    print("\nDetailed Per-Class Performance for Option A (4-Class):")
    for name, m in results_a.items():
        print(f"\n--- Model: {name} ---")
        print(m["classification_report"])

    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON TABLE: OPTION B (3-CLASS: SHOCK EXCLUDED)")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for name, m in results_b.items():
        print(f"{name:<20} | {m['balanced_accuracy']*100:>12.2f}% | {m['macro_precision']*100:>12.2f}% | {m['macro_recall']*100:>12.2f}% | {m['macro_f1']*100:>8.2f}%")
    print("=" * 90)

    print("\nDetailed Per-Class Performance for Option B (3-Class):")
    for name, m in results_b.items():
        print(f"\n--- Model: {name} ---")
        print(m["classification_report"])


def export_latex_table(results_a: dict, results_b: dict):
    lines = [
        "% Auto-generated by train_classifiers.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Edge Classifier Benchmark Comparison Across Physical Regimes (5-Fold StratifiedGroupKFold OOF)}",
        "\\label{tab:model_benchmarks}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Model Architecture} & \\textbf{Balanced Acc} & \\textbf{Macro Prec} & \\textbf{Macro Rec} & \\textbf{Macro-F1} \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textbf{Option A: 4-Class Kinematics (Baseline, Tilt, Shock, Buffeting)}} \\\\",
        "\\midrule",
    ]

    best_a_name = max(results_a.keys(), key=lambda k: results_a[k]["macro_f1"])
    for name, m in results_a.items():
        b_acc = m["balanced_accuracy"] * 100.0
        prec = m["macro_precision"] * 100.0
        rec = m["macro_recall"] * 100.0
        f1 = m["macro_f1"] * 100.0
        if name == best_a_name:
            lines.append(f"    \\textbf{{{name}}} & \\textbf{{{b_acc:.2f}\\%}} & \\textbf{{{prec:.2f}\\%}} & \\textbf{{{rec:.2f}\\%}} & \\textbf{{{f1:.2f}\\%}} \\\\")
        else:
            lines.append(f"    {name} & {b_acc:.2f}\\% & {prec:.2f}\\% & {rec:.2f}\\% & {f1:.2f}\\% \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textbf{Option B: 3-Class Kinematics (Baseline, Tilt, Buffeting)}} \\\\")
    lines.append("\\midrule")

    best_b_name = max(results_b.keys(), key=lambda k: results_b[k]["macro_f1"])
    for name, m in results_b.items():
        b_acc = m["balanced_accuracy"] * 100.0
        prec = m["macro_precision"] * 100.0
        rec = m["macro_recall"] * 100.0
        f1 = m["macro_f1"] * 100.0
        if name == best_b_name:
            lines.append(f"    \\textbf{{{name}}} & \\textbf{{{b_acc:.2f}\\%}} & \\textbf{{{prec:.2f}\\%}} & \\textbf{{{rec:.2f}\\%}} & \\textbf{{{f1:.2f}\\%}} \\\\")
        else:
            lines.append(f"    {name} & {b_acc:.2f}\\% & {prec:.2f}\\% & {rec:.2f}\\% & {f1:.2f}\\% \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(LATEX_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[OK] Updated LaTeX benchmark table: {LATEX_TABLE_PATH}")


def plot_dual_confusion_matrices(results_a: dict, results_b: dict):
    """
    Renders an 8-panel high-resolution (300 DPI) comparison figure:
      Row 0: Option A (4-Class): Linear SVM, Random Forest, Edge MLP, and Performance Bar Chart
      Row 1: Option B (3-Class): Linear SVM, Random Forest, Edge MLP, and Performance Bar Chart
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    model_names = ["Linear SVM", "Random Forest", "Edge MLP"]
    cmaps_a = ["Blues", "Greens", "Purples"]
    cmaps_b = ["Blues", "Greens", "Purples"]

    # --- ROW 0: Option A (4-Class) ---
    for i, name in enumerate(model_names):
        ax = axes[0, i]
        cm = results_a[name]["confusion_matrix"]
        b_acc = results_a[name]["balanced_accuracy"] * 100.0
        f1 = results_a[name]["macro_f1"] * 100.0

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmaps_a[i],
            ax=ax,
            xticklabels=DISPLAY_NAMES_A,
            yticklabels=DISPLAY_NAMES_A,
            cbar=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(
            f"(A{i+1}) Option A: {name}\nBal Acc: {b_acc:.2f}% | Macro-F1: {f1:.2f}%",
            fontweight="bold",
            fontsize=11,
            pad=8,
        )
        ax.set_xlabel("Predicted Physical Regime", fontweight="bold", fontsize=9)
        ax.set_ylabel("True Physical Regime", fontweight="bold", fontsize=9)
        ax.tick_params(axis="x", rotation=25)

    # Subplot A4: Summary Bar Chart for Option A
    ax_bar_a = axes[0, 3]
    b_accs_a = [results_a[m]["balanced_accuracy"] * 100.0 for m in model_names]
    f1s_a = [results_a[m]["macro_f1"] * 100.0 for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35

    ax_bar_a.bar(x - width / 2, b_accs_a, width, label="Balanced Acc (%)", color="#2b5c8f")
    ax_bar_a.bar(x + width / 2, f1s_a, width, label="Macro-F1 (%)", color="#d95f02")
    ax_bar_a.set_xticks(x)
    ax_bar_a.set_xticklabels(model_names, rotation=20, ha="right", fontweight="bold", fontsize=9)
    ax_bar_a.set_ylabel("Score (%)", fontweight="bold", fontsize=9)
    ax_bar_a.set_title("(A4) Option A: Model Comparison", fontweight="bold", fontsize=11, pad=8)
    ax_bar_a.set_ylim(50, 95)
    ax_bar_a.grid(axis="y", linestyle="--", alpha=0.7)
    ax_bar_a.legend(loc="upper left", frameon=True, fontsize=8)

    # --- ROW 1: Option B (3-Class) ---
    for i, name in enumerate(model_names):
        ax = axes[1, i]
        cm = results_b[name]["confusion_matrix"]
        b_acc = results_b[name]["balanced_accuracy"] * 100.0
        f1 = results_b[name]["macro_f1"] * 100.0

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmaps_b[i],
            ax=ax,
            xticklabels=DISPLAY_NAMES_B,
            yticklabels=DISPLAY_NAMES_B,
            cbar=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(
            f"(B{i+1}) Option B: {name}\nBal Acc: {b_acc:.2f}% | Macro-F1: {f1:.2f}%",
            fontweight="bold",
            fontsize=11,
            pad=8,
        )
        ax.set_xlabel("Predicted Physical Regime", fontweight="bold", fontsize=9)
        ax.set_ylabel("True Physical Regime", fontweight="bold", fontsize=9)
        ax.tick_params(axis="x", rotation=25)

    # Subplot B4: Summary Bar Chart for Option B
    ax_bar_b = axes[1, 3]
    b_accs_b = [results_b[m]["balanced_accuracy"] * 100.0 for m in model_names]
    f1s_b = [results_b[m]["macro_f1"] * 100.0 for m in model_names]

    ax_bar_b.bar(x - width / 2, b_accs_b, width, label="Balanced Acc (%)", color="#2b5c8f")
    ax_bar_b.bar(x + width / 2, f1s_b, width, label="Macro-F1 (%)", color="#d95f02")
    ax_bar_b.set_xticks(x)
    ax_bar_b.set_xticklabels(model_names, rotation=20, ha="right", fontweight="bold", fontsize=9)
    ax_bar_b.set_ylabel("Score (%)", fontweight="bold", fontsize=9)
    ax_bar_b.set_title("(B4) Option B: Model Comparison", fontweight="bold", fontsize=11, pad=8)
    ax_bar_b.set_ylim(50, 95)
    ax_bar_b.grid(axis="y", linestyle="--", alpha=0.7)
    ax_bar_b.legend(loc="upper left", frameon=True, fontsize=8)

    plt.suptitle(
        "Out-of-Fold (OOF) Benchmark Across Edge Architectures: 4-Class (Temporal Block Grouping) vs 3-Class Kinematics",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=300)
    plt.close()
    print(f"\n[OK] Saved dual 8-panel confusion matrix figure (300 DPI): {CM_PLOT_PATH}")


def serialize_model_weights(best_name: str, best_clf, best_scaler, best_metrics: dict, class_names: list, display_names: list, option_label: str):
    export_dict = {
        "metadata": {
            "option": option_label,
            "model_name": best_name,
            "macro_f1": float(best_metrics["macro_f1"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "n_features": len(FEATURE_COLS),
            "n_classes": len(class_names),
            "feature_names": FEATURE_COLS,
            "class_names": class_names,
            "display_names": display_names,
        },
        "scaler": {
            "mean": [float(m) for m in best_scaler.mean_],
            "scale": [float(s) for s in best_scaler.scale_],
            "var": [float(v) for v in best_scaler.var_],
        },
        "model_parameters": {},
    }

    if isinstance(best_clf, RandomForestClassifier):
        export_dict["model_parameters"]["type"] = "RandomForestClassifier"
        export_dict["model_parameters"]["n_estimators"] = len(best_clf.estimators_)
        export_dict["model_parameters"]["max_depth"] = best_clf.max_depth
        trees = []
        for idx, est in enumerate(best_clf.estimators_):
            tree = est.tree_
            trees.append({
                "tree_id": idx,
                "node_count": int(tree.node_count),
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": [round(float(t), 6) for t in tree.threshold],
                "values": [v[0].tolist() for v in tree.value],
            })
        export_dict["model_parameters"]["trees"] = trees

    elif isinstance(best_clf, LinearSVC):
        export_dict["model_parameters"]["type"] = "LinearSVC"
        export_dict["model_parameters"]["classes"] = [int(c) for c in best_clf.classes_]
        export_dict["model_parameters"]["coefficients"] = [
            [round(float(c), 6) for c in row] for row in best_clf.coef_
        ]
        export_dict["model_parameters"]["intercepts"] = [
            round(float(b), 6) for b in best_clf.intercept_
        ]

    elif isinstance(best_clf, MLPClassifier):
        export_dict["model_parameters"]["type"] = "MLPClassifier"
        export_dict["model_parameters"]["hidden_layer_sizes"] = list(best_clf.hidden_layer_sizes)
        export_dict["model_parameters"]["activation"] = best_clf.activation
        export_dict["model_parameters"]["weights"] = [
            [[round(float(w), 6) for w in col] for col in mat] for mat in best_clf.coefs_
        ]
        export_dict["model_parameters"]["biases"] = [
            [round(float(b), 6) for b in vec] for vec in best_clf.intercepts_
        ]

    return export_dict


def export_models(results_a: dict, fitted_a: dict, results_b: dict, fitted_b: dict):
    # Option A Winner (Primary 4-class deployment)
    best_a_name = max(results_a.keys(), key=lambda k: results_a[k]["macro_f1"])
    best_a_clf, best_a_scaler = fitted_a[best_a_name]
    best_a_metrics = results_a[best_a_name]

    print("\n" + "=" * 80)
    print(f"OPTION A (4-CLASS) BEST PERFORMING MODEL: {best_a_name}")
    print(f" Macro-F1 Score    : {best_a_metrics['macro_f1']:.4f}")
    print(f" Balanced Accuracy : {best_a_metrics['balanced_accuracy']:.4f}")
    print("=" * 80)

    dict_a = serialize_model_weights(
        best_a_name, best_a_clf, best_a_scaler, best_a_metrics,
        CLASS_NAMES_A, DISPLAY_NAMES_A, "Option A (4-Class)"
    )
    with open(WEIGHTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dict_a, f, indent=2)
    size_a_kb = os.path.getsize(WEIGHTS_JSON_PATH) / 1024.0
    print(f"[OK] Exported Option A best model weights: {WEIGHTS_JSON_PATH} ({size_a_kb:.1f} KB)")

    # Option B Winner (Alternative 3-class deployment)
    best_b_name = max(results_b.keys(), key=lambda k: results_b[k]["macro_f1"])
    best_b_clf, best_b_scaler = fitted_b[best_b_name]
    best_b_metrics = results_b[best_b_name]

    print("\n" + "=" * 80)
    print(f"OPTION B (3-CLASS) BEST PERFORMING MODEL: {best_b_name}")
    print(f" Macro-F1 Score    : {best_b_metrics['macro_f1']:.4f}")
    print(f" Balanced Accuracy : {best_b_metrics['balanced_accuracy']:.4f}")
    print("=" * 80)

    dict_b = serialize_model_weights(
        best_b_name, best_b_clf, best_b_scaler, best_b_metrics,
        CLASS_NAMES_B, DISPLAY_NAMES_B, "Option B (3-Class)"
    )
    with open(WEIGHTS_OPT_B_PATH, "w", encoding="utf-8") as f:
        json.dump(dict_b, f, indent=2)
    size_b_kb = os.path.getsize(WEIGHTS_OPT_B_PATH) / 1024.0
    print(f"[OK] Exported Option B best model weights: {WEIGHTS_OPT_B_PATH} ({size_b_kb:.1f} KB)")


def main():
    df_a, df_b = prepare_datasets()

    # 1. Option A: 4-Class Kinematic Benchmark
    res_a, oof_a, fit_a, X_a, y_a = run_cv_benchmark(
        df_a, "Option A (4-Class)", CLASS_NAMES_A, DISPLAY_NAMES_A
    )

    # 2. Option B: 3-Class Kinematic Benchmark
    res_b, oof_b, fit_b, X_b, y_b = run_cv_benchmark(
        df_b, "Option B (3-Class)", CLASS_NAMES_B, DISPLAY_NAMES_B
    )

    # 3. Print Comparison Tables & Per-Class Reports
    print_comparison_tables(res_a, res_b)

    # 4. Export LaTeX Table
    export_latex_table(res_a, res_b)

    # 5. Render Dual 8-Panel Visual Comparison
    plot_dual_confusion_matrices(res_a, res_b)

    # 6. Export Best Model Weights (JSON)
    export_models(res_a, fit_a, res_b, fit_b)


if __name__ == "__main__":
    main()
