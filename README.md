# 🎥 Real-Time Video Anomaly Detection System

A **computer vision–based real-time anomaly detection system** that analyzes live camera feeds or video files to identify **suspicious human activities** such as abnormal motion, restricted zone intrusion, fighting behavior, panic events, and unusual movement patterns. The system uses **OpenCV**, **centroid-based object tracking**, and **behavior analysis** to generate alerts, logs, and visual evidence.

---

## 🚀 Project Overview

This project processes video streams in real time to:

* Detect and track people
* Classify human behavior (walking, running, sitting, falling, idle)
* Identify security anomalies
* Log events with timestamps and bounding boxes
* Trigger alerts with visual and audio feedback

It is suitable for **smart surveillance**, **home security**, **campus monitoring**, and **public safety systems**.

---

## ✨ Key Features

### 🧍 Human Detection & Tracking

* Uses **HOG + SVM people detector**
* Centroid-based multi-object tracking
* Assigns persistent IDs to individuals

### 🧠 Behavior Classification

* Walking
* Running
* Sitting
* Falling (improved height-change logic)
* Idle detection

### 🚨 Anomaly Detection Events

* **Motion Alerts** (background subtraction based)
* **Restricted Zone Intrusion**
* **Fight Detection** (proximity + movement patterns)
* **Panic Mode Alerts** (manual trigger)

### 🔐 Restricted Zone Monitoring

* Configurable on-screen restricted area
* Entry/exit detection
* Automatic snapshot capture

### 👨‍👩‍👧 Family Mode

* Filters alerts to only critical events
* Reduces false alarms in home environments

### 📸 Evidence Collection

* Saves annotated frames for each alert
* Logs all events into CSV format

### 🔊 Audio Alerts

* Cross-platform alert sound support (Windows/Linux/macOS)

---

## 🛠 Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV
* **Tracking:** Centroid Tracker
* **Math & Data:** NumPy, SciPy, Pandas
* **UI / Display:** OpenCV GUI, Tkinter (screen handling)
* **Platform Support:** Windows, Linux, macOS

---

## 📁 Project Structure

```
project-root/
│
├── Anomaly_detector.py     # Main application
├── output/                # Event logs & snapshots (auto-created)
│   ├── events_log.csv
│   └── *.jpg
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/video-anomaly-detector.git
cd video-anomaly-detector
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python numpy pandas scipy imutils
```

> Optional (Windows): `winsound` is used automatically for alert sounds.

---

## ▶️ Usage

### Run with Webcam

```bash
python Anomaly_detector.py
```

### Run with Video File

```bash
python Anomaly_detector.py --video path/to/video.mp4
```

### Custom Output Directory

```bash
python Anomaly_detector.py --out-dir results
```

---

## ⌨️ Keyboard Controls

| Key | Action                 |
| --- | ---------------------- |
| `f` | Toggle Family Mode     |
| `p` | Trigger Panic Alert    |
| `r` | Toggle Restricted Zone |
| `q` | Quit Application       |

---

## 📊 Event Logging

All detected events are logged in:

```
output/events_log.csv
```

### Logged Fields:

* Timestamp
* Event Type
* Object ID
* Frame Number
* Bounding Box
* Detected Behavior

---

## 🧪 Detection Logic Summary

* **Motion Detection:** Background subtraction (MOG2)
* **Tracking:** Euclidean distance–based centroid matching
* **Fight Detection:** Close proximity + aggressive movement
* **Fall Detection:** Sudden height reduction + motion spike

---

## ⚠️ Limitations

* Designed for single-camera environments
* Performance depends on lighting and camera angle
* HOG detector may struggle in crowded scenes
* Not intended for medical or legal decision-making

---

## 🔮 Future Enhancements

* Deep learning–based person detection (YOLO / SSD)
* Pose estimation for better action recognition
* Multi-camera support
* Cloud-based alert notifications
* Web dashboard for event analytics

---

## 👨‍💻 Author

**Prince Harsha**
Computer Vision & Security Systems Developer

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you find this project useful, feel free to star the repository!
