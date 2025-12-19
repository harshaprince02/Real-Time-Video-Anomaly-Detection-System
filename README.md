# Real-Time-Video-Anomaly-Detection-System
Developed a real-time video anomaly detection system using Python and OpenCV to monitor human activities. Implemented centroid-based tracking, motion analysis, and behavior classification (walking, running, sitting, falling) to detect restricted-zone intrusions, fights, panic events, and abnormal motion, with event logging and alerts.

Real-Time Video Anomaly Detection System – How to Run
Prerequisites

Python 3.8 or higher

A webcam or a recorded video file

OS: Windows / macOS / Linux

Required Libraries

Install the dependencies using:

pip install opencv-python numpy pandas scipy imutils


Note (Windows users):
winsound is built-in and does not require installation.

Running with Webcam (Live Detection)
python Anomaly_detector.py

Running with a Video File
python Anomaly_detector.py --video path/to/video.mp4

Optional Command-Line Arguments

--out-dir output → Directory to store logs and snapshots

--family-mode → Filters alerts to only critical events

--motion-thresh-percent 0.5 → Motion sensitivity threshold

Example:

python Anomaly_detector.py --video sample.mp4 --family-mode --out-dir results

Keyboard Controls During Execution

F → Toggle Family Mode

P → Trigger Panic Alert

R → Enable/Disable Restricted Zone

Q → Quit Application

Outputs

Event logs saved as CSV files

Snapshot images for alerts (zone intrusion, fight, panic)

Real-time on-screen alerts and audible notifications
