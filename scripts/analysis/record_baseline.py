import serial
import csv
import time
import sys
import numpy as np

SERIAL_PORT = 'COM3'  
BAUD_RATE = 115200
OUTPUT_FILE = 'mpu6050_unstable_ruler_Zaxis_High.csv'
DURATION_SECONDS = 300

STALE_SAMPLE_LIMIT = 20  # Stop if sensor repeats identical readings for 20 frames (~200ms)

print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(1)
    ser.reset_input_buffer()
except Exception as e:
    print(f"Error opening {SERIAL_PORT}: {e}")
    sys.exit(1)

print(f"\n[RECORDING STARTED] Writing to {OUTPUT_FILE} for {DURATION_SECONDS}s")
print("Keep the sensor completely flat and motionless on the bench.\n")

headers = ['timestamp_ms', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
samples = []
start_time = time.time()

last_imu_reading = None
stale_count = 0
aborted_due_to_freeze = False

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    while (time.time() - start_time) < DURATION_SECONDS:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) == 7:
                vals = [int(p) for p in parts]
                current_imu = vals[1:]  # [ax, ay, az, gx, gy, gz]

                # ==========================================
                # STALE / FROZEN SENSOR GUARD
                # ==========================================
                if current_imu == last_imu_reading:
                    stale_count += 1
                    if stale_count >= STALE_SAMPLE_LIMIT:
                        print(f"\n\n[FATAL ERROR] I2C Bus Freeze Detected!")
                        print(f"Repeated identical reading {stale_count} times: {current_imu}")
                        print("Terminating recording to prevent corrupted dataset.")
                        aborted_due_to_freeze = True
                        break
                else:
                    stale_count = 0
                    last_imu_reading = current_imu

                writer.writerow(vals)
                samples.append(current_imu)

                if len(samples) % 50 == 0:
                    remaining = int(DURATION_SECONDS - (time.time() - start_time))
                    print(f"Logged {len(samples)} samples... ({remaining}s left)", end='\r')

        except KeyboardInterrupt:
            print("\nRecording stopped early by user.")
            break
        except Exception:
            continue

ser.close()

if aborted_due_to_freeze:
    print(f"\nRun failed. Partial rows logged: {len(samples)}")
    print("Check DuPont wire connections, then rerun.")
    sys.exit(1)

print(f"\nFinished! Total rows recorded: {len(samples)}")

if samples:
    data = np.array(samples)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    print("\n--- Calibration Results ---")
    print(f"Accel X: Mean={means[0]:>8.2f} | Std={stds[0]:.2f}")
    print(f"Accel Y: Mean={means[1]:>8.2f} | Std={stds[1]:.2f}")
    print(f"Accel Z: Mean={means[2]:>8.2f} | Std={stds[2]:.2f}")
    print(f"Gyro  X: Mean={means[3]:>8.2f} | Std={stds[3]:.2f}")
    print(f"Gyro  Y: Mean={means[4]:>8.2f} | Std={stds[4]:.2f}")
    print(f"Gyro  Z: Mean={means[5]:>8.2f} | Std={stds[5]:.2f}")