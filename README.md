# Real-Time Traffic Counter & Vehicle Tracking using YOLOv8

A computer vision pipeline built in Python to perform real-time vehicle detection, tracking, and line-crossing counting on highway traffic video feeds.

![Demo](/.demo.gif)

Key Features
- Object Detection: Powered by `YOLOv8n` pre-trained on the COCO dataset.
- Vehicle Filtering:Isolates vehicle classes (cars, motorcycles, buses, trucks).
- Object Tracking: Employs `ByteTrack` to maintain persistent vehicle IDs across frames.
- Line Crossing Counter: Dynamically tracks directional vehicle counts using Supervision `LineZone`.

Tech Stack
- Language: Python 3.14
- Libraries: OpenCV, Ultralytics YOLOv8, Supervision, NumPy

Quick Start
1. Install requirements:
   ```bash
   pip install ultralytics opencv-python supervision numpy
