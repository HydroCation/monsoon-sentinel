# 🌧️ Project: The Monsoon Sentinel
### Littoral Edge Intelligence Node for Monsoon Dynamics

The **Monsoon Sentinel** is a self-contained, edge-intelligent sensing node designed to study the coupled atmospheric–oceanic dynamics of the Northeast Monsoon in the Bay of Bengal. Instead of acting as a passive data logger, the system performs on-device, physics-informed feature extraction, transmitting only high-value "events of interest" to conserve energy and bandwidth in harsh littoral environments.

---

## 1. System Philosophy & Architecture

The node processes data locally across three specialized sensing layers, optimizing for edge intelligence and minimal data transmission.

### 📡 The Sensing Stack

* **Acoustic Layer:** Uses an **INMP441** digital microphone via **I2S** as a passive sonar system. It handles rain-rate characterization and tracks maritime "noise floor" anomalies such as vessel traffic.
* **Kinetic Layer:** Uses an **MPU-6050** 6-axis IMU via **I2C** for motion tracking. It drives wave-period analysis and sea-state estimation.
* **Atmospheric Layer:** Uses a custom **long-wire antenna** routed to an internal **ADC** to detect Very Low Frequency (VLF, 3–30 kHz) "Sferics". This tracks lightning propagation through the Earth-Ionosphere waveguide.

### 🧠 The Compute Strategy

* **Platform:** Powered by an **STM32F401RE** (ARM Cortex-M4 @ 84MHz) utilizing its hardware Floating Point Unit (FPU).
* **Local DSP:** Leverages the **CMSIS-DSP** library to compute real-time Fast Fourier Transforms (FFTs) and extract Mel-Frequency Cepstral Coefficients (MFCCs) from high-rate sensor streams.
* **Inference:** Employs an on-device classifier to filter out background noise and isolate "Events of Interest" (e.g., squall onset vs. background swell).

---

## 2. Hardware Mapping & Protocol Blueprint

The system splits communication across distinct interfaces, using a shared I2C bus for motion and telemetry, alongside high-speed digital and analog sampling pipelines.

| Component | Protocol | Nucleo-F401RE Pins | Primary Debug Tool |
| :--- | :--- | :--- | :--- |
| **INMP441** | I2S | `PA4` (WS), `PA5` (CK), `PA7` (SD) | nanoDLA (Digital Analysis)  |
| **MPU-6050** | I2C | `PB8` (SCL), `PB9` (SDA) | PulseView (Timing Validation)  |
| **OLED (SSH1106)** | I2C | Shared Bus (`PB8` / `PB9`) | Visual Status Output  |
| **Long-Wire Antenna** | Analog | `PA0` (ADC1_IN0) | Oscilloscope (Signal Integrity)  |

---

## 3. Physical & Strategic Constraints

Operating a deployment node in open littoral waters presents unique survivability challenges:

* **The Marine Environment:** The hardware must survive conductive saltwater (which acts as a heavy RF shield), persistent high humidity, and pressure shifts. The node is deployed within a custom, pressure-sealed PVC housing.
* **Power Autonomy:** The node is shifting from a development-phase tethered USB connection to an autonomous Li-Ion battery system. It utilizes strict low-power "Sleep/Wake" interrupts triggered by the IMU or acoustic thresholds to maximize longevity.
* **The Physics Hypothesis:** The project explores a correlation between localized VLF pulse distortion and monsoon-driven humidity gradients within the boundary layer.

---

## 4. Repository Directory Structure

```text
monsoon-sentinel/
├── .gitignore               # Excludes compiler and IDE noise
├── README.md                # Project overview and specifications
├── Makefile                 # Build automation script
├── monsoon-sentinel.ioc     # STM32CubeMX Hardware configuration file
├── STM32F401XX_FLASH.ld     # Linker Script
├── startup_stm32f401xe.s    # Hardware startup assembly
├── docs/                    # Architectural and inventory documentation
│   ├── DESIGN_VISION.md
│   └── INVENTORY.md
├── Core/                    # Core Application Code
│   ├── Inc/                 # Header files (main.h, interrupts)
│   └── Src/                 # Application source (main.c, DSP code)
└── Drivers/                 # ST Hardware Abstraction Layer & CMSIS
