#!/usr/bin/env python3
"""
Hardened Host Serial Telemetry Ingestion Script for Monsoon Sentinel.
Decouples serial I/O from disk writes using a producer-consumer threading model.
"""

import argparse
import csv
import os
import queue
import sys
import threading
import time
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None

STALE_SAMPLE_LIMIT = 20  # Warn if sensor repeats identical readings for 20 frames (~200ms)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hardened Serial Telemetry Logger for STM32 + MPU-6050"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port interface (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (default: 115200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/telemetry_capture.csv",
        help="Output CSV path (default: data/raw/telemetry_capture.csv)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Logging duration in seconds (default: 300)",
    )
    return parser.parse_args()


def serial_reader_thread(ser, data_queue, stop_event):
    """
    Dedicated worker thread for high-priority serial I/O.
    Prevents OS serial RX buffer overflow by immediately enqueueing incoming lines.
    """
    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                data_queue.put(line)
        except Exception as e:
            if not stop_event.is_set():
                # Report intermittent serial read error
                pass


def main():
    args = parse_args()

    if serial is None:
        print("[ERROR] pyserial is not installed. Install via: pip install pyserial")
        sys.exit(1)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MONSOON SENTINEL - RESILIENT TELEMETRY LOGGER")
    print(f" Port      : {args.port}")
    print(f" Baud      : {args.baud}")
    print(f" Output    : {output_path}")
    print(f" Duration  : {args.duration} s")
    print("=" * 70)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        time.sleep(1.0)  # Settle connection
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[ERROR] Failed to open serial port {args.port}: {e}")
        sys.exit(1)

    data_queue = queue.Queue(maxsize=10000)
    stop_event = threading.Event()

    reader_thread = threading.Thread(
        target=serial_reader_thread,
        args=(ser, data_queue, stop_event),
        daemon=True,
    )
    reader_thread.start()

    headers = [
        "timestamp_ms",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]

    last_imu_reading = None
    stale_count = 0
    total_stale_events = 0
    total_samples = 0
    start_time = time.time()

    print("\n[INFO] Ingestion pipeline started. Press Ctrl+C to stop.\n")

    try:
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            while (time.time() - start_time) < args.duration and not stop_event.is_set():
                try:
                    line = data_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                parts = line.split(",")
                if len(parts) == 7:
                    try:
                        vals = [int(p.strip()) for p in parts]
                    except ValueError:
                        continue

                    current_imu = tuple(vals[1:])  # (ax, ay, az, gx, gy, gz)

                    # Stale / Frozen frame detection
                    if current_imu == last_imu_reading:
                        stale_count += 1
                        if stale_count == STALE_SAMPLE_LIMIT:
                            total_stale_events += 1
                            elapsed = time.time() - start_time
                            print(
                                f"\n[WARN @ {elapsed:0.1f}s] Sensor freeze detected! "
                                f"Repeated identical reading {stale_count} times: {current_imu}. "
                                "Continuing capture without aborting..."
                            )
                        elif stale_count > STALE_SAMPLE_LIMIT and stale_count % 100 == 0:
                            elapsed = time.time() - start_time
                            print(
                                f"\n[WARN @ {elapsed:0.1f}s] Sensor STILL frozen: "
                                f"{stale_count} consecutive identical frames."
                            )
                    else:
                        stale_count = 0
                        last_imu_reading = current_imu

                    writer.writerow(vals)
                    total_samples += 1

                    if total_samples % 100 == 0:
                        elapsed = time.time() - start_time
                        rem = max(0, int(args.duration - elapsed))
                        q_size = data_queue.qsize()
                        print(
                            f"Logged {total_samples:>6} samples | Queue: {q_size:>3} | Remaining: {rem:>3}s",
                            end="\r",
                            flush=True,
                        )

    except KeyboardInterrupt:
        print("\n\n[INFO] Capture interrupted by user.")
    finally:
        stop_event.set()
        reader_thread.join(timeout=1.0)
        try:
            ser.close()
        except Exception:
            pass

    actual_duration = time.time() - start_time
    avg_fs = total_samples / actual_duration if actual_duration > 0 else 0.0

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print(f" Total Samples Logged : {total_samples}")
    print(f" Elapsed Duration      : {actual_duration:.2f} s")
    print(f" Effective Sample Rate : {avg_fs:.2f} Hz")
    print(f" Stale/Freeze Warnings : {total_stale_events}")
    print(f" Saved Destination     : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
