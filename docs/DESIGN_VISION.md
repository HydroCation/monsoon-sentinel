# 🌧️ Project: The Monsoon Sentinel
### Littoral Edge Intelligence Node for Monsoon Dynamics

---

## 0. Vision

The Monsoon Sentinel is a **self-contained, edge-intelligent sensing node** designed to study the coupled atmospheric–oceanic dynamics of the Northeast Monsoon in the Bay of Bengal.

Instead of acting as a passive data logger, the system performs **on-device, physics-informed feature extraction**, transmitting only high-value “events of interest”.

> Goal: Build a resilient, low-power, deployable node that survives harsh littoral conditions while generating scientifically meaningful signals.

---

## 1. System Philosophy

- **Edge > Cloud**  
  Process locally, transmit minimally.

- **Physics-Informed Sensing**  
  Extract meaningful features (spectral, temporal, statistical), not raw streams.

- **Survivability First**  
  Mechanical + power reliability is more critical than algorithmic complexity.

- **Iterative Deployment**  
  Lab → Controlled Outdoor → Coastal → Open Water

---

## 2. Current Hardware Inventory

### 🧠 Compute Core
- STM32 Nucleo-F401RE (Cortex-M4F @ 84 MHz)
- 96 KB SRAM / 512 KB Flash
- Hardware FPU (critical for DSP)

### 🎧 Sensors
- INMP441 (I2S digital microphone)
- MPU-6050 (6-axis IMU)

### 📟 Output / Debug
- SSH1106-based 128x64 OLED (I2C)

### 🔬 Instrumentation
- nanoDLA Logic Analyzer (24 MHz, 8-channel)

### 🔧 Prototyping
- Breadboard + jumper wires

---

## 3. System Architecture

### 3.1 Sensing Stack

#### Acoustic Layer (Primary)
- Device: INMP441 (I2S)
- Function:
  - Rain rate estimation (spectral density)
  - Ambient maritime noise profiling
- Output:
  - FFT bins
  - MFCC features

#### Kinetic Layer
- Device: MPU-6050 (I2C)
- Function:
  - Wave period estimation
  - Sea state classification
- Output:
  - Time-domain motion signatures
  - Frequency-domain wave energy

#### Atmospheric Layer (Planned)
- Device: Long-wire antenna + ADC
- Function:
  - VLF (3–30 kHz) sferic detection
- Output:
  - Pulse timing + distortion metrics

---

### 3.2 Compute Pipeline

#### Stage 1: Acquisition
- DMA-driven sampling (I2S + I2C polling/interrupts)

#### Stage 2: DSP
- CMSIS-DSP:
  - FFT (acoustic + motion)
  - Windowing + filtering
- Feature Extraction:
  - MFCC (audio)
  - Spectral energy bands
  - Peak detection

#### Stage 3: Inference
- Lightweight classifier:
  - Rule-based (initial)
  - TinyML (future)

**Target Events:**
- Rain onset
- Squall signatures
- Vessel proximity
- Abnormal wave patterns

---

### 3.3 Power Strategy

#### Current State
- USB-powered (development phase)

#### Target State
- Li-Ion battery system

#### Design Principles
- Sleep-first architecture
- Interrupt-driven wake:
  - IMU motion trigger
  - Acoustic threshold trigger

#### Power Modes
- Deep Sleep → Monitoring
- Active → DSP + inference
- Burst → Data logging/transmission

---

### 3.4 Data Strategy

- Raw data is **not stored continuously**
- Only store:
  - Feature vectors
  - Event summaries
  - Short buffered snapshots (if needed)

---

## 4. Mechanical & Environmental Design (Critical)

### Challenges
- Saltwater corrosion
- High humidity / condensation
- Pressure sealing

### Planned Solutions
- PVC pressure housing
- Conformal coating / epoxy potting
- Cable gland sealing
- Internal desiccant

> ⚠️ Mechanical failure is the highest risk in the system.

---

## 5. Communication Strategy (Open Problem)

### Constraints
- RF attenuation underwater
- Power limitations

### Candidate Approaches
- Surface buoy relay (preferred)
- Store-and-retrieve logging
- Acoustic signaling (future)

---

## 6. The Physics Hypothesis

> Investigate correlation between **VLF sferic distortion** and **humidity gradients during monsoon transitions**.

### Rationale
- Lightning signals propagate via Earth-ionosphere waveguide
- Humidity + atmospheric structure may influence signal characteristics

### Required Outputs
- Pulse shape metrics
- Frequency dispersion
- Temporal clustering

---

## 7. Development Roadmap

### Phase 1 — Lab Validation
- I2S microphone + FFT working
- IMU data acquisition stable
- OLED debug output

### Phase 2 — Feature Extraction
- MFCC pipeline
- Wave period estimation
- Event detection (rule-based)

### Phase 3 — Edge Intelligence
- TinyML classifier
- Event tagging system

### Phase 4 — Power Autonomy
- Battery integration
- Sleep/wake system

### Phase 5 — Field Prototype
- Waterproof enclosure
- Short-duration deployment

---

## 8. Risks & Unknowns

| Area | Risk |
|------|------|
| Mechanical sealing | High |
| Power longevity | Medium |
| Memory constraints | Medium |
| VLF signal clarity | High |
| Communication | Undefined |

---

## 9. Design Principles (Non-Negotiable)

- No feature without measurable signal value
- No continuous streaming unless justified
- Always test in real-world noise conditions
- Prefer robustness over complexity

---

## 10. Living Notes

- OLED uses **SSH1106**, not SSD1306 (driver differences matter)
- I2S + DMA stability is critical before adding AI
- Logic analyzer is mandatory for debugging timing issues

---

## 11. Future Extensions

- Multi-node distributed sensing network
- Coastal data assimilation
- Integration with meteorological datasets
- Collaboration with research institutes

---

## 12. Identity

**The Monsoon Sentinel is not a gadget.**  
It is a **field-deployable scientific instrument** built under constraint.

---