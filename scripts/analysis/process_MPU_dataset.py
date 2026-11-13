import numpy as np
import pandas as pd
import os
from pathlib import Path

# ==========================================
# PATH RESOLUTION & DIRECTORY SETUP
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "master_features_v4.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CONSTANTS & SCALING
# ==========================================
FS = 100.0                   # 100 Hz sampling rate
WINDOW_SIZE = 128            # 1.28s window (2^7 for CMSIS-DSP FFT)
STEP_SIZE = 64               # 50% overlap (0.64s step)
EPSILON = 1e-7               # Numerical safety guard

ACCEL_SCALE = 16384.0        # LSB/g (+/- 2g)
GYRO_SCALE  = 131.0          # LSB/(deg/s) (+/- 250 dps)

# Exact non-overlapping mapping: Filename -> (Regime_Name, Class_ID)
FILE_REGIME_MAP = {
    # 0: Nominal Baseline (Flat, Fan, Suspended Breeze, Stationary Ruler)
    "mpu6050_baseline_stationary.csv":           ("Nominal_Baseline", 0),
    "mpu6050_baseline_stationary_2.csv":         ("Nominal_Baseline", 0),
    "mpu6050_baseline_stationary_3.csv":         ("Nominal_Baseline", 0),
    "mpu6050_baseline_stationary_4.csv":         ("Nominal_Baseline", 0),
    "mpu6050_baseline_stationary_room_fan_1.csv":("Nominal_Baseline", 0),
    "mpu6050_15_min_data_fan_1.csv":             ("Nominal_Baseline", 0),
    "mpu6050_baseline_ambient_breeze_5min.csv":  ("Nominal_Baseline", 0),
    "mpu6050_baseline_ambient_breeze_30min.csv": ("Nominal_Baseline", 0),
    "mpu6050_unstable_ruler_3.csv":              ("Nominal_Baseline", 0),
    "mpu6050_unstable_ruler_Xaxis_1.csv":        ("Nominal_Baseline", 0),

    # 1: Structural Tilt
    "mpu6050_minor_tilt_1.csv":                  ("Structural_Tilt", 1),
    "mpu6050_medium_tilt_1.csv":                 ("Structural_Tilt", 1),
    "mpu6050_high_tilt_1.csv":                   ("Structural_Tilt", 1),

    # 2: Acoustic Vibration
    "mpu6050_400Hz.csv":                         ("Acoustic_Vibration", 2),
    "mpu6050_400Hz_2.csv":                       ("Acoustic_Vibration", 2),

    # 3: Impulse Shock
    "mpu6050_flat_taps_1.csv":                   ("Impulse_Shock", 3),

    # 4: Dynamic Structural Buffeting / Wind Excitation (Mast Buffeting)
    "mpu6050_unstable_ruler_light.csv":             ("Mast_Buffeting", 4),
    "mpu6050_unstable_ruler_medium_faceon.csv":     ("Mast_Buffeting", 4),
    "mpu6050_unstable_ruler_heavy_faceon.csv":      ("Mast_Buffeting", 4),
    "mpu6050_unstable_ruler_Xaxis_FAN_Medium.csv":  ("Mast_Buffeting", 4),
    "mpu6050_unstable_ruler_Xaxis_FAN_High.csv":    ("Mast_Buffeting", 4),
    "mpu6050_unstable_ruler_Xaxis_FAN_Ceiling.csv": ("Mast_Buffeting", 4),
}

def extract_features(window_raw):
    """
    Orientation-invariant, physically scaled feature extractor.
    window_raw: (128, 6) [ax, ay, az, gx, gy, gz]
    """
    # 1. Physical Units
    ax_g = window_raw[:, 0] / ACCEL_SCALE
    ay_g = window_raw[:, 1] / ACCEL_SCALE
    az_g = window_raw[:, 2] / ACCEL_SCALE

    gx_dps = window_raw[:, 3] / GYRO_SCALE
    gy_dps = window_raw[:, 4] / GYRO_SCALE
    gz_dps = window_raw[:, 5] / GYRO_SCALE

    # 2. Static Gravity Vector & Euler Orientation
    mean_ax = np.mean(ax_g)
    mean_ay = np.mean(ay_g)
    mean_az = np.mean(az_g)
    static_g = np.sqrt(mean_ax**2 + mean_ay**2 + mean_az**2)

    pitch_deg = float(np.degrees(np.arctan2(-mean_ax, np.sqrt(mean_ay**2 + mean_az**2) + EPSILON)))
    roll_deg  = float(np.degrees(np.arctan2(mean_ay, mean_az + EPSILON)))
    cos_tilt  = np.clip(np.abs(mean_az) / (static_g + EPSILON), -1.0, 1.0)
    tilt_deg  = float(np.degrees(np.arccos(cos_tilt)))

    # 3. 3D Dynamic Acceleration Norm (Tilt-Invariant)
    dyn_x = ax_g - mean_ax
    dyn_y = ay_g - mean_ay
    dyn_z = az_g - mean_az
    a_dyn = np.sqrt(dyn_x**2 + dyn_y**2 + dyn_z**2)

    # 4. Time-Domain Metrics
    var_dyn   = float(np.var(a_dyn))
    rms_dyn   = float(np.sqrt(np.mean(a_dyn**2)))
    p2p_dyn   = float(np.ptp(a_dyn))
    peak_dyn  = float(np.max(a_dyn))
    crest_factor = float(peak_dyn / (rms_dyn + EPSILON))
    gyro_rms  = float(np.sqrt(np.mean(gx_dps**2 + gy_dps**2 + gz_dps**2)))

    # 5. Frequency-Domain (CMSIS-DSP RFFT Compatible)
    hanning = np.hanning(WINDOW_SIZE)
    a_dyn_centered = a_dyn - np.mean(a_dyn)
    fft_mag = np.abs(np.fft.rfft(a_dyn_centered * hanning))
    fft_freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/FS)

    peak_idx = np.argmax(fft_mag[1:]) + 1
    dominant_freq = float(fft_freqs[peak_idx])

    band_low  = float(np.sum(fft_mag[(fft_freqs >= 0.5) & (fft_freqs < 3.0)]**2))
    band_mid  = float(np.sum(fft_mag[(fft_freqs >= 3.0) & (fft_freqs < 12.0)]**2))
    band_high = float(np.sum(fft_mag[(fft_freqs >= 12.0)]**2))
    total_energy = band_low + band_mid + band_high + EPSILON
    spectral_ratio_low = float(band_low / total_energy)

    return [
        tilt_deg, pitch_deg, roll_deg,
        var_dyn, rms_dyn, p2p_dyn, crest_factor,
        gyro_rms, dominant_freq,
        band_low, band_mid, band_high, spectral_ratio_low
    ]

# ==========================================
# PROCESSING PIPELINE
# ==========================================
feature_cols = [
    'tilt_deg', 'pitch_deg', 'roll_deg',
    'var_dyn_g2', 'rms_dyn_g', 'p2p_dyn_g', 'crest_factor',
    'gyro_rms_dps', 'dominant_freq_hz',
    'band_low', 'band_mid', 'band_high', 'spectral_ratio_low',
    'source_file', 'regime_name', 'label'
]

def run_pipeline():
    rows = []
    processed_count = 0

    print("="*75)
    print(f"Executing Feature Extraction Pipeline: raw -> {OUTPUT_FILE.name}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DIR}")
    print("="*75)

    for filename, (regime, label_id) in FILE_REGIME_MAP.items():
        filepath = RAW_DIR / filename
        if not filepath.exists():
            print(f" [!] Missing file: {filepath}")
            continue

        df = pd.read_csv(filepath)

        # Data Cleansing: mpu6050_unstable_ruler_Xaxis_FAN_Ceiling.csv
        # Drop row 0 (which has a timestamp of 4020ms before a 430s stall) and reset timestamps
        if filename == "mpu6050_unstable_ruler_Xaxis_FAN_Ceiling.csv":
            initial_len = len(df)
            df = df.iloc[1:].reset_index(drop=True)
            if 'timestamp_ms' in df.columns:
                df['timestamp_ms'] = df['timestamp_ms'] - df['timestamp_ms'].iloc[0]
            print(f" [Cleansed] {filename:<43} | Dropped row 0 (stall) | {initial_len} -> {len(df)} rows")

        if 'accel_x' in df.columns:
            raw_mat = df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        else:
            raw_mat = df.iloc[:, 1:7].values if df.shape[1] >= 7 else df.iloc[:, :6].values

        num_w = (len(raw_mat) - WINDOW_SIZE) // STEP_SIZE + 1
        for w in range(num_w):
            s = w * STEP_SIZE
            e = s + WINDOW_SIZE
            feats = extract_features(raw_mat[s:e])
            rows.append(feats + [filename, regime, label_id])

        print(f" -> {filename:<45} | {num_w:>4} windows | {regime} (ID {label_id})")
        processed_count += 1

    master_df = pd.DataFrame(rows, columns=feature_cols)
    master_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*75)
    print(f"EXTRACTION COMPLETE: {processed_count} files processed | {len(master_df)} total feature vectors")
    print(f"Saved: {OUTPUT_FILE}")
    print("="*75)

    print("\nClass Distribution:")
    vc = master_df['regime_name'].value_counts()
    for reg, count in vc.items():
        pct = count / len(master_df) * 100
        print(f"  {reg:<22} : {count:>5} windows ({pct:>5.2f}%)")

    print("\nLabel Distribution:")
    lvc = master_df['label'].value_counts().sort_index()
    for lid, count in lvc.items():
        pct = count / len(master_df) * 100
        print(f"  Label {lid:<2} : {count:>5} windows ({pct:>5.2f}%)")

    print("\nFeature Summary Across 5 Regimes (SI Units Mean):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    summary_cols = ['tilt_deg', 'rms_dyn_g', 'p2p_dyn_g', 'crest_factor', 'gyro_rms_dps', 'dominant_freq_hz']
    print(master_df.groupby('regime_name')[summary_cols].mean().reindex([
        'Nominal_Baseline',
        'Structural_Tilt',
        'Acoustic_Vibration',
        'Impulse_Shock',
        'Mast_Buffeting'
    ]))

if __name__ == "__main__":
    run_pipeline()
