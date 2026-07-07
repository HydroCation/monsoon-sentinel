# 🌧️ Project: The Monsoon Sentinel
### Littoral Edge Intelligence Node for Monsoon Dynamics

[cite_start]The **Monsoon Sentinel** is a self-contained, edge-intelligent sensing node designed to study the coupled atmospheric–oceanic dynamics of the Northeast Monsoon in the Bay of Bengal[cite: 1]. Instead of acting as a passive data logger, the system performs on-device, physics-informed feature extraction, transmitting only high-value "events of interest" to conserve energy and bandwidth in harsh littoral environments.

---

## 1. System Philosophy & Architecture

[cite_start]The node processes data locally across three specialized sensing layers, optimizing for edge intelligence and minimal data transmission[cite: 7].

### 📡 The Sensing Stack

* [cite_start]**Acoustic Layer:** Uses an **INMP441** digital microphone via **I2S** as a passive sonar system[cite: 2, 11]. [cite_start]It handles rain-rate characterization and tracks maritime "noise floor" anomalies such as vessel traffic[cite: 2].
* [cite_start]**Kinetic Layer:** Uses an **MPU-6050** 6-axis IMU via **I2C** for motion tracking[cite: 4, 11]. [cite_start]It drives wave-period analysis and sea-state estimation[cite: 4].
* [cite_start]**Atmospheric Layer:** Uses a custom **long-wire antenna** routed to an internal **ADC** to detect Very Low Frequency (VLF, 3–30 kHz) "Sferics"[cite: 3, 11]. [cite_start]This tracks lightning propagation through the Earth-Ionosphere waveguide[cite: 3].

### 🧠 The Compute Strategy

* [cite_start]**Platform:** Powered by an **STM32F401RE** (ARM Cortex-M4 @ 84MHz) utilizing its hardware Floating Point Unit (FPU)[cite: 5].
* [cite_start]**Local DSP:** Leverages the **CMSIS-DSP** library to compute real-time Fast Fourier Transforms (FFTs) and extract Mel-Frequency Cepstral Coefficients (MFCCs) from high-rate sensor streams[cite: 6].
* [cite_start]**Inference:** Employs an on-device classifier to filter out background noise and isolate "Events of Interest" (e.g., squall onset vs. background swell)[cite: 7].

---

## 2. Hardware Mapping & Protocol Blueprint

The system splits communication across distinct interfaces, using a shared I2C bus for motion and telemetry, alongside high-speed digital and analog sampling pipelines.

| Component | Protocol | Nucleo-F401RE Pins | Primary Debug Tool |
| :--- | :--- | :--- | :--- |
| **INMP441** | I2S | [cite_start]`PA4` (WS), `PA5` (CK), `PA7` (SD) | nanoDLA (Digital Analysis) [cite: 11] |
| **MPU-6050** | I2C | `PB8` (SCL), `PB9` (SDA) | [cite_start]PulseView (Timing Validation) [cite: 11] |
| **OLED (SSH1106)** | I2C | Shared Bus (`PB8` / `PB9`) | [cite_start]Visual Status Output [cite: 11] |
| **Long-Wire Antenna** | Analog | `PA0` (ADC1_IN0) | [cite_start]Oscilloscope (Signal Integrity) [cite: 11] |

---

## 3. Physical & Strategic Constraints

Operating a deployment node in open littoral waters presents unique survivability challenges:

* [cite_start]**The Marine Environment:** The hardware must survive conductive saltwater (which acts as a heavy RF shield), persistent high humidity, and pressure shifts[cite: 8]. [cite_start]The node is deployed within a custom, pressure-sealed PVC housing[cite: 8].
* [cite_start]**Power Autonomy:** The node is shifting from a development-phase tethered USB connection to an autonomous Li-Ion battery system[cite: 9]. [cite_start]It utilizes strict low-power "Sleep/Wake" interrupts triggered by the IMU or acoustic thresholds to maximize longevity[cite: 9].
* [cite_start]**The Physics Hypothesis:** The project explores a correlation between localized VLF pulse distortion and monsoon-driven humidity gradients within the boundary layer[cite: 10].

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