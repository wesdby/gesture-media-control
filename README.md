# 🖐 Gesture Media Control

A lightweight macOS utility that allows you to control your media playback (**Spotify** and **Apple Music**) and system volume using intuitive hand gestures captured via your webcam. 

Powered by **OpenCV**, **MediaPipe**, and a custom implementation of the **One Euro Filter** for ultra-smooth tracking, featuring a native **macOS Menu Bar** interface.

---

## Features

* **Touchless Control:** Adjust volume, skip tracks, play/pause, and scrub through timelines without touching your keyboard.
* **Advanced Smoothing:** Integrated **One Euro Filter** eliminates jitter and lag, providing butter-smooth wrist and hand tracking.
* **Smart Gestures with Hysteresis:** Robust threshold detection for precise activation and zero accidental triggers.
* **macOS Native Integration:** Runs quietly in the **Menu Bar** via `rumps` with a live optional camera preview window.
* **Auto-Recovery:** Automatic webcam reconnection handling if the camera feed drops or freezes (e.g., after wake from sleep).

---

## 🦾 Supported Gestures

|.         Gesture         |        Action      |                                                 Description                                                    | 
| :--- | :--- | :--- |
| **Point (Index Finger)** | **Volume Control** | Rotate your index finger in a circular motion (clockwise / counter-clockwise) to raise or lower system volume. |
| **Pinch (Thumb + Index)** | **Timeline Seeking** | Pinch and drag horizontally left/right to scrub backward or forward through the track. |
| **Open Palm (Swipe)** | **Track Navigation** | Swipe horizontally to switch tracks (Next/Previous) or vertically to Play/Pause. |
| **Fist (Hold for 1s)** | **Toggle Tracking** | Hold a closed fist for 1 second to temporarily pause or resume gesture tracking. |

---

## Requirements & Limitations

* **Operating System:** **macOS ONLY** (relies on AppleScript for media control and native system volume hooks).
* **Hardware:** A working webcam (built-in or external).
* **Python Version:** Python 3.8 – 3.12 recommended.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/wesdby/gesture-media-control.git](https://github.com/wesdby/gesture-media-control.git)
   cd gesture-media-control
