# LAB INVENTORY: Signal Processing & AI Station

## 1. The Core (Compute)
### **STM32 Nucleo-F401RE**
* **MCU:** STM32F401RET6 (ARM Cortex-M4F)
* **Clock Speed:** 84 MHz
* **Memory:** 512 KB Flash / 96 KB SRAM
* **FPU:** Single-precision Floating Point Unit (Hardware Math)
* **Debugger:** On-board ST-LINK/V2-1 (switched to VCOM/Mass Storage)
* **Voltage:** 3.3V Logic (5V Tolerant I/O)
* **Key Peripherals:** I2S (Audio), I2C (Sensors), SPI, USART.

---

## 2. The Sensors (Input)
### **INMP441 MEMS Microphone**
* **Type:** Omnidirectional MEMS
* **Interface:** **I2S** (Inter-IC Sound) - *Digital Output*
* **Specs:** 24-bit data, High Signal-to-Noise Ratio (SNR)
* **Voltage:** 3.3V
* **Purpose:** High-fidelity audio capture for Keyword Spotting (AI).

### **MPU-6050 IMU**
* **Type:** 6-Axis Motion Tracking
* **Sensors:** 3-Axis Gyroscope + 3-Axis Accelerometer
* **Interface:** **I2C**
* **ADC:** 16-bit internal ADCs
* **Purpose:** Gesture recognition (Magic Wand), vibration analysis.

---

## 3. The Feedback (Output)
### **GoldenMorning 1.3" OLED Display**
* **Driver Chip:** **SSH1106** (Note: Not SSD1306)
* **Resolution:** 128 x 64 pixels
* **Interface:** **I2C** (4-pin: VCC, GND, SDA, SCL)
* **Color:** White
* **Voltage:** 3.3V - 5V

---

## 4. The Debugger (Instrumentation)
### **Muse Lab nanoDLA Logic Analyzer**
* **Sample Rate:** 24 MHz (Max)
* **Channels:** 8 Digital Inputs
* **Connection:** USB Type-C
* **Software:** Sigrok / PulseView (Open Source)
* **Purpose:** Visualizing I2C/I2S timing, debugging "Invisible" protocol errors.

---

## 5. Prototyping Gear
* **Breadboard:** MB102 (830 Points)
* **Wires:**
    * **M-M:** Board-to-Board / Breadboard internal routing.
    * **M-F:** Breadboard-to-Nucleo (Arduino Headers).
    * **F-F:** Nucleo (Morpho Pins)-to-Logic Analyzer / Sensor direct.