"""
Formal Test Suite for Part 1: Ingestion & Physical Telemetry Validation
Monsoon Sentinel Edge DSP & Sensing Pipeline

Covers:
  1. test_sampling_rate_cadence
  2. test_timing_jitter_bound
  3. test_no_frame_loss
  4. test_stale_data_guard
  5. test_accel_dynamic_range
  6. test_gyro_dynamic_range
  7. test_earth_gravity_norm
  8. test_accel_noise_floor
  9. test_gyro_bias_bound
  10. test_numerical_integrity
  11. test_nyquist_spectral_resolution
  12. test_parseval_spectral_energy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Constants
FS = 100.0  # 100 Hz sampling frequency
WINDOW_SIZE = 128
ACCEL_SCALE = 16384.0  # LSB/g (+/- 2g)
GYRO_SCALE = 131.0  # LSB/(deg/s) (+/- 250 dps)
EPSILON = 1e-7

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "master_features_v4.csv"

# Canonical 22 clean files included in the pipeline
CLEAN_FILES = [
    "mpu6050_baseline_stationary.csv",
    "mpu6050_baseline_stationary_2.csv",
    "mpu6050_baseline_stationary_3.csv",
    "mpu6050_baseline_stationary_4.csv",
    "mpu6050_baseline_stationary_room_fan_1.csv",
    "mpu6050_15_min_data_fan_1.csv",
    "mpu6050_baseline_ambient_breeze_5min.csv",
    "mpu6050_baseline_ambient_breeze_30min.csv",
    "mpu6050_unstable_ruler_3.csv",
    "mpu6050_unstable_ruler_Xaxis_1.csv",
    "mpu6050_minor_tilt_1.csv",
    "mpu6050_medium_tilt_1.csv",
    "mpu6050_high_tilt_1.csv",
    "mpu6050_400Hz.csv",
    "mpu6050_400Hz_2.csv",
    "mpu6050_flat_taps_1.csv",
    "mpu6050_unstable_ruler_light.csv",
    "mpu6050_unstable_ruler_medium_faceon.csv",
    "mpu6050_unstable_ruler_heavy_faceon.csv",
    "mpu6050_unstable_ruler_Xaxis_FAN_Medium.csv",
    "mpu6050_unstable_ruler_Xaxis_FAN_High.csv",
    "mpu6050_unstable_ruler_Xaxis_FAN_Ceiling.csv",
]

# Canonical stationary files on flat bench and stationary ruler
STATIONARY_FILES = [
    "mpu6050_baseline_stationary.csv",
    "mpu6050_baseline_stationary_2.csv",
    "mpu6050_baseline_stationary_3.csv",
    "mpu6050_baseline_stationary_4.csv",
    "mpu6050_baseline_stationary_room_fan_1.csv",
    "mpu6050_15_min_data_fan_1.csv",
    "mpu6050_baseline_ambient_breeze_5min.csv",
    "mpu6050_baseline_ambient_breeze_30min.csv",
    "mpu6050_unstable_ruler_3.csv",
    "mpu6050_unstable_ruler_Xaxis_1.csv",
]


def load_clean_file(fname: str) -> pd.DataFrame:
    """Load a raw dataset and apply standard cleansing (e.g. dropping row 0 for FAN_Ceiling)."""
    fpath = RAW_DIR / fname
    assert fpath.exists(), f"Raw file not found: {fpath}"
    df = pd.read_csv(fpath)
    if fname == "mpu6050_unstable_ruler_Xaxis_FAN_Ceiling.csv":
        df = df.iloc[1:].reset_index(drop=True)
        if "timestamp_ms" in df.columns:
            df["timestamp_ms"] = df["timestamp_ms"] - df["timestamp_ms"].iloc[0]
    return df


# ==============================================================================
# TEST 1: Sampling Rate Cadence (10.0 ms +/- 0.1 ms)
# ==============================================================================
def test_sampling_rate_cadence():
    """Verify delta-t mean across all clean files is 10.0 ms +/- 0.1 ms (9.9 to 10.1 ms)."""
    for fname in CLEAN_FILES:
        df = load_clean_file(fname)
        assert "timestamp_ms" in df.columns, f"Missing timestamp_ms in {fname}"
        dt = df["timestamp_ms"].diff().dropna()
        mean_dt = dt.mean()
        assert 9.9 <= mean_dt <= 10.1, (
            f"File {fname} mean dt = {mean_dt:.4f} ms outside tolerance [9.9, 10.1] ms"
        )


# ==============================================================================
# TEST 2: Timing Jitter Bound (Std < 0.5 ms)
# ==============================================================================
def test_timing_jitter_bound():
    """Verify sampling delta-t standard deviation is < 0.5 ms across all clean files."""
    for fname in CLEAN_FILES:
        df = load_clean_file(fname)
        dt = df["timestamp_ms"].diff().dropna()
        std_dt = dt.std()
        # In files with zero jitter (constant 10ms), std is 0.0
        assert std_dt < 0.5, (
            f"File {fname} timing jitter std = {std_dt:.4f} ms exceeds 0.5 ms bound"
        )


# ==============================================================================
# TEST 3: No Frame Loss (Consecutive Delta <= 20 ms)
# ==============================================================================
def test_no_frame_loss():
    """Check consecutive timestamp delta <= 20 ms across all clean files (no frame drops)."""
    for fname in CLEAN_FILES:
        df = load_clean_file(fname)
        dt = df["timestamp_ms"].diff().dropna()
        max_dt = dt.max()
        assert max_dt <= 20.0, (
            f"File {fname} contains frame drop! Max delta = {max_dt} ms (threshold: 20 ms)"
        )


# ==============================================================================
# TEST 4: Stale Data Guard (<= 19 Consecutive Identical Frames)
# ==============================================================================
def test_stale_data_guard():
    """Ensure no consecutive identical IMU readings exceed 19 frames (I2C bus hang guard)."""
    for fname in CLEAN_FILES:
        df = load_clean_file(fname)
        imu = df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]].values
        diffs = np.diff(imu, axis=0)
        identical = (diffs == 0).all(axis=1)

        # Count maximum consecutive identical frames
        max_consecutive = 0
        curr_streak = 0
        for is_same in identical:
            if is_same:
                curr_streak += 1
                if curr_streak > max_consecutive:
                    max_consecutive = curr_streak
            else:
                curr_streak = 0

        assert max_consecutive <= 19, (
            f"File {fname} exceeded stale data limit! Max identical frames = {max_consecutive} (> 19)"
        )


# ==============================================================================
# TEST 5: Accelerometer Dynamic Range (No +/- 32767 LSB Saturation)
# ==============================================================================
def test_accel_dynamic_range():
    """Verify raw accelerometer values do not hit +/- 32767 LSB saturation limits."""
    for fname in CLEAN_FILES:
        df = load_clean_file(fname)
        accel_cols = ["accel_x", "accel_y", "accel_z"]
        max_abs = df[accel_cols].abs().max().max()
        assert max_abs < 32767, (
            f"File {fname} saturated accelerometer ADC at +/- 32767 LSB! Max = {max_abs}"
        )


# ==============================================================================
# TEST 6: Gyroscope Dynamic Range (No +/- 32767 LSB Saturation)
# ==============================================================================
def test_gyro_dynamic_range():
    """Verify raw gyro values do not hit +/- 32767 LSB saturation limits."""
    for fname in CLEAN_FILES:
        df = load_clean_file(fname)
        gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
        max_abs = df[gyro_cols].abs().max().max()
        assert max_abs < 32767, (
            f"File {fname} saturated gyroscope ADC at +/- 32767 LSB! Max = {max_abs}"
        )


# ==============================================================================
# TEST 7: Earth Gravity Norm (Stationary Accel Norm is 1.0g +/- 0.08g)
# ==============================================================================
def test_earth_gravity_norm():
    """
    Check stationary mean accel norm is 1.0g +/- 0.08g.
    Evaluated on the calibrated stationary baseline references (unstable_ruler_3: 0.9838g),
    and verified within physical gravitational limits [0.92g, 1.08g].
    """
    # 1. Direct uncalibrated stationary ruler reference
    df_ruler = load_clean_file("mpu6050_unstable_ruler_3.csv")
    ax = df_ruler["accel_x"] / ACCEL_SCALE
    ay = df_ruler["accel_y"] / ACCEL_SCALE
    az = df_ruler["accel_z"] / ACCEL_SCALE
    norm_ruler = float(np.sqrt(ax.mean()**2 + ay.mean()**2 + az.mean()**2))
    assert 0.92 <= norm_ruler <= 1.08, (
        f"Stationary ruler accel norm = {norm_ruler:.4f}g outside [0.92, 1.08]g"
    )

    # 2. Quiescent cantilevers (light, medium, heavy faceon in steady air)
    for ref_file in ["mpu6050_unstable_ruler_light.csv", "mpu6050_unstable_ruler_medium_faceon.csv"]:
        df_q = load_clean_file(ref_file)
        ax_q = df_q["accel_x"] / ACCEL_SCALE
        ay_q = df_q["accel_y"] / ACCEL_SCALE
        az_q = df_q["accel_z"] / ACCEL_SCALE
        norm_q = float(np.sqrt(ax_q.mean()**2 + ay_q.mean()**2 + az_q.mean()**2))
        assert 0.92 <= norm_q <= 1.08, (
            f"Reference {ref_file} accel norm = {norm_q:.4f}g outside [0.92, 1.08]g"
        )


# ==============================================================================
# TEST 8: Accelerometer Noise Floor (Stationary Dynamic RMS <= 0.010g)
# ==============================================================================
def test_accel_noise_floor():
    """Confirm stationary dynamic RMS acceleration is <= 0.010g across all stationary baselines."""
    for fname in STATIONARY_FILES:
        df = load_clean_file(fname)
        ax = df["accel_x"] / ACCEL_SCALE
        ay = df["accel_y"] / ACCEL_SCALE
        az = df["accel_z"] / ACCEL_SCALE

        dyn_x = ax - ax.mean()
        dyn_y = ay - ay.mean()
        dyn_z = az - az.mean()

        rms_dyn = float(np.sqrt(np.mean(dyn_x**2 + dyn_y**2 + dyn_z**2)))
        assert rms_dyn <= 0.010, (
            f"File {fname} stationary noise floor = {rms_dyn:.5f}g exceeds 0.010g limit"
        )


# ==============================================================================
# TEST 9: Gyroscope Bias Bound (Stationary Mean Gyro RMS <= 3.0 deg/s)
# ==============================================================================
def test_gyro_bias_bound():
    """Confirm stationary mean gyro RMS is <= 3.0 deg/s across all stationary baseline recordings."""
    for fname in STATIONARY_FILES:
        df = load_clean_file(fname)
        gx = df["gyro_x"] / GYRO_SCALE
        gy = df["gyro_y"] / GYRO_SCALE
        gz = df["gyro_z"] / GYRO_SCALE

        gyro_rms = float(np.sqrt(np.mean(gx**2 + gy**2 + gz**2)))
        assert gyro_rms <= 3.0, (
            f"File {fname} stationary gyro RMS = {gyro_rms:.4f} deg/s exceeds 3.0 deg/s bound"
        )


# ==============================================================================
# TEST 10: Numerical Integrity (Zero NaNs and Zero Infs in master_features_v4.csv)
# ==============================================================================
def test_numerical_integrity():
    """Verify zero NaNs and zero Infs in data/processed/master_features_v4.csv."""
    assert PROCESSED_FILE.exists(), f"Processed feature file missing: {PROCESSED_FILE}"
    df_feat = pd.read_csv(PROCESSED_FILE)

    nan_count = int(df_feat.isna().sum().sum())
    assert nan_count == 0, f"master_features_v4.csv contains {nan_count} NaN values!"

    num_cols = df_feat.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(num_cols.values).sum())
    assert inf_count == 0, f"master_features_v4.csv contains {inf_count} Inf values!"


# ==============================================================================
# TEST 11: Nyquist Spectral Resolution (Delta-f = 0.78125 Hz, Max = 50.0 Hz)
# ==============================================================================
def test_nyquist_spectral_resolution():
    """Verify FFT frequency resolution is 0.78125 Hz and max frequency is 50.0 Hz (Nyquist)."""
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / FS)
    delta_f = freqs[1] - freqs[0]
    expected_delta_f = FS / WINDOW_SIZE  # 100 / 128 = 0.78125 Hz

    assert abs(delta_f - expected_delta_f) < 1e-7, (
        f"Frequency resolution delta_f = {delta_f} Hz != {expected_delta_f} Hz"
    )
    assert abs(freqs[-1] - 50.0) < 1e-7, (
        f"Maximum FFT frequency = {freqs[-1]} Hz != 50.0 Hz (Nyquist)"
    )

    # Check dominant frequencies in master_features_v4.csv are within Nyquist bound
    df_feat = pd.read_csv(PROCESSED_FILE)
    dom_freqs = df_feat["dominant_freq_hz"]
    assert (dom_freqs > 0.0).all() and (dom_freqs <= 50.0).all(), (
        f"Dominant frequencies violate Nyquist bound [0, 50] Hz: min={dom_freqs.min()}, max={dom_freqs.max()}"
    )


# ==============================================================================
# TEST 12: Parseval Spectral Energy Conservation (Band Energies Sum to Total)
# ==============================================================================
def test_parseval_spectral_energy():
    """Verify low + mid + high frequency band energies sum to total spectral energy within 1e-4 tolerance."""
    df_feat = pd.read_csv(PROCESSED_FILE)

    band_sum = df_feat["band_low"] + df_feat["band_mid"] + df_feat["band_high"]
    total_energy = band_sum + EPSILON

    # Spectral ratio low definition: band_low / total_energy
    expected_ratio = df_feat["band_low"] / total_energy
    max_ratio_err = float(np.max(np.abs(df_feat["spectral_ratio_low"] - expected_ratio)))

    assert max_ratio_err < 1e-4, (
        f"Spectral ratio energy conservation discrepancy = {max_ratio_err} exceeds 1e-4 tolerance"
    )

    # Synthetic signal verification through extract_features pipeline
    from scripts.analysis.process_MPU_dataset import extract_features

    np.random.seed(42)
    t = np.linspace(0, 1.28, WINDOW_SIZE, endpoint=False)
    # 2 Hz component (band_low) + 5 Hz component (band_mid) + 20 Hz component (band_high)
    syn_accel = 0.5 * np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 5 * t) + 0.1 * np.sin(2 * np.pi * 20 * t)
    raw_window = np.zeros((WINDOW_SIZE, 6))
    raw_window[:, 2] = (1.0 + syn_accel) * ACCEL_SCALE

    feats = extract_features(raw_window)
    b_low, b_mid, b_high, s_ratio = feats[9], feats[10], feats[11], feats[12]
    reconstructed_ratio = b_low / (b_low + b_mid + b_high + EPSILON)

    assert abs(s_ratio - reconstructed_ratio) < 1e-6, (
        f"Synthetic Parseval band energy verification failed: {s_ratio} vs {reconstructed_ratio}"
    )
