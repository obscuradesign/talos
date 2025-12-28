![TALOS Road Test 1](https://github.com/user-attachments/assets/4817b9d0-cf50-4279-b368-3cef3c574246)

# TALOS: Bicycle Safety System (V1.5)
**Headless AI Computer Vision for Raspberry Pi 5**

TALOS is an open-source safety system that uses a Raspberry Pi 5 and a Global Shutter Camera Module to detect vehicles approaching cyclists from behind. It broadcasts a live video feed with object detection to the rider's phone via a private WiFi hotspot.

Why? Real-time blind spot detection for Vulnerable Road Users (VRU) designed for low-cost micromobility fleets.

## Features
* **Auto-Start:** Runs automatically 10 seconds after boot (Headless Mode).
* **Private Network:** Creates `TALOS_LINK` hotspot for off-grid connection.
* **High-Speed AI:** Uses fine-tuned version of YOLOv8 (NCNN optimized) for ~10 FPS detection on Pi 5.
* **Hardware:** Raspberry Pi 5 + IMX296 Global Shutter Camera.

## Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/obscuradesign/TALOS.git](https://github.com/obscuradesign/TALOS.git)
    ```
2.  Install dependencies (Ultralytics, NCNN, Flask).
3.  Enable the systemd service for auto-start.

## Usage
1.  Power on the Raspberry Pi.
2.  Connect your phone to the `TALOS_LINK` WiFi network.
3.  Open a browser and go to `http://10.42.0.1:5000`.

---
*Created by Kevin Davidson / Obscura Design*
