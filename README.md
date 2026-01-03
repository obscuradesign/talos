# TALOS: Bicycle Safety System (V2.0 - Hailo Edition)

**Hardware-Accelerated AI Computer Vision for Raspberry Pi 5**

TALOS is an open-source safety system that uses a Raspberry Pi 5, a Hailo-8L NPU, and a Global Shutter Camera Module to detect vehicles approaching cyclists from behind. It broadcasts a live video feed with object detection to the rider's phone via a private WiFi hotspot.

> **⚠️ Hardware Requirement:** This branch is optimized for the **Hailo-8L AI Accelerator**. If you are running on a standard Raspberry Pi 5 without the NPU, please use the **[legacy-cpu](https://github.com/obscuradesign/talos/tree/legacy-cpu)** branch.

### Features

* **Auto-Start:** Runs automatically 10 seconds after boot (Headless Mode).
* **Private Network:** Creates `TALOS_LINK` hotspot for off-grid connection.
* **Hardware-Accelerated AI:** Utilizes the Hailo-8L NPU to run compiled `.hef` models, offloading the CPU and providing ultra-low latency inference.
* **Hardware:** Raspberry Pi 5 + Hailo-8L AI Hat+ + IMX296 Global Shutter Camera.
* **Collision Warning Logic:** Real-time Time-To-Collision (TTC) estimation triggers visual alerts when vehicles approach < 4 seconds. (Coming soon)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/obscuradesign/talos.git](https://github.com/obscuradesign/talos.git)
