import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 0. DIRECTORY & ENVIRONMENT SETUP
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DATA = PROJECT_ROOT / "data" / "processed" / "master_features_v4.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

if not INPUT_DATA.exists():
    raise FileNotFoundError(f"Input feature file not found at: {INPUT_DATA}")

df = pd.read_csv(INPUT_DATA)

# Explicit palette for all 5 physical regimes
color_map = {
    'Nominal_Baseline':   '#2b5c8f',     # Muted Navy
    'Structural_Tilt':    '#d95f02',     # Rust Orange
    'Acoustic_Vibration': '#1b9e77',     # Teal Green
    'Impulse_Shock':      '#e7298a',     # Magenta
    'Mast_Buffeting':     '#7570b3'      # Royal Purple
}

# Preserve strict categorical ordering in plots
REGIME_ORDER = [
    'Nominal_Baseline',
    'Structural_Tilt',
    'Acoustic_Vibration',
    'Impulse_Shock',
    'Mast_Buffeting'
]
df['regime_name'] = pd.Categorical(df['regime_name'], categories=REGIME_ORDER, ordered=True)

# =====================================================================
# 1. MULTI-PANEL 2D SCATTER SEPARABILITY
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))

# Panel A: Static Tilt vs Dynamic Energy
for regime in REGIME_ORDER:
    grp = df[df['regime_name'] == regime]
    if grp.empty:
        continue
    axes[0].scatter(
        grp['tilt_deg'], grp['rms_dyn_g'],
        label=regime.replace('_', ' '), color=color_map.get(regime, 'gray'),
        alpha=0.45, s=22, edgecolors='none'
    )
axes[0].set_xlabel("Static Tilt Angle (°)", fontweight='bold')
axes[0].set_ylabel("Dynamic RMS Acceleration (g)", fontweight='bold')
axes[0].set_title("A: Static Tilt vs. Dynamic Linear Energy", fontweight='bold')
axes[0].axvline(5.0, color='gray', linestyle='--', alpha=0.7, label='Tilt Threshold (~5°)')
axes[0].legend(loc='upper right', frameon=True, fontsize=8)

# Panel B: Impulse Shock Space (Peak-to-Peak vs Crest Factor)
for regime in REGIME_ORDER:
    grp = df[df['regime_name'] == regime]
    if grp.empty:
        continue
    axes[1].scatter(
        grp['crest_factor'], grp['p2p_dyn_g'],
        label=regime.replace('_', ' '), color=color_map.get(regime, 'gray'),
        alpha=0.45, s=22, edgecolors='none'
    )
axes[1].set_xlabel("Crest Factor (Peak / RMS)", fontweight='bold')
axes[1].set_ylabel("Peak-to-Peak Accel (g)", fontweight='bold')
axes[1].set_title("B: Transient Impulse / Shock Space", fontweight='bold')
axes[1].set_yscale('log')
axes[1].legend(loc='upper left', frameon=True, fontsize=8)

# Panel C: Rotational Energy vs Structural Vibration (Isolates Mast Buffeting)
for regime in REGIME_ORDER:
    grp = df[df['regime_name'] == regime]
    if grp.empty:
        continue
    axes[2].scatter(
        grp['gyro_rms_dps'], grp['rms_dyn_g'],
        label=regime.replace('_', ' '), color=color_map.get(regime, 'gray'),
        alpha=0.45, s=22, edgecolors='none'
    )
axes[2].set_xlabel("Gyro Total RMS (°/s)", fontweight='bold')
axes[2].set_ylabel("Dynamic RMS Acceleration (g)", fontweight='bold')
axes[2].set_title("C: Angular Rate vs. Linear Vibration", fontweight='bold')
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].legend(loc='lower right', frameon=True, fontsize=8)

plt.tight_layout()
scatter_path = OUTPUT_DIR / "eda_scatter_separability.png"
plt.savefig(scatter_path, dpi=300)
print(f"Saved: {scatter_path}")
plt.close()

# =====================================================================
# 2. KEY FEATURE DISTRIBUTION BOXPLOTS (5 CRITICAL DIMENSIONS)
# =====================================================================
fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))

feature_targets = [
    ('tilt_deg', 'Static Tilt (°)', False),
    ('rms_dyn_g', 'Dynamic RMS (g)', True),
    ('p2p_dyn_g', 'Peak-to-Peak Accel (g)', True),
    ('crest_factor', 'Crest Factor (Unitless)', False),
    ('gyro_rms_dps', 'Gyro RMS (°/s)', True)
]

for ax, (col, label, use_log) in zip(axes, feature_targets):
    sns.boxplot(
        data=df, 
        x='regime_name', 
        y=col, 
        order=REGIME_ORDER,
        hue='regime_name', 
        legend=False, 
        ax=ax, 
        palette=color_map, 
        showmeans=True, 
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5}
    )
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=35)
    if use_log:
        ax.set_yscale('log')

plt.tight_layout()
box_path = OUTPUT_DIR / "eda_boxplots_distributions.png"
plt.savefig(box_path, dpi=300)
print(f"Saved: {box_path}")
plt.close()

# =====================================================================
# 3. COLLINEARITY & FEATURE CORRELATION HEATMAP
# =====================================================================
feature_cols = [
    'tilt_deg', 'pitch_deg', 'roll_deg',
    'var_dyn_g2', 'rms_dyn_g', 'p2p_dyn_g', 'crest_factor',
    'gyro_rms_dps', 'dominant_freq_hz',
    'band_low', 'band_mid', 'band_high', 'spectral_ratio_low'
]

plt.figure(figsize=(11, 9))
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, cmap='vlag', vmin=-1.0, vmax=1.0,
    annot=True, fmt=".2f", square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Physical Feature Pearson Correlation Matrix (5 Regimes)", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
corr_path = OUTPUT_DIR / "eda_feature_correlation.png"
plt.savefig(corr_path, dpi=300)
print(f"Saved: {corr_path}")
plt.close()

print(f"\nAll 5-regime EDA plots successfully generated at 300 DPI in: {OUTPUT_DIR}")
