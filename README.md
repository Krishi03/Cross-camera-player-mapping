# Cross-Camera Player Detection and Tracking

## Overview

This project provides a pipeline for **detecting, tracking, and matching players across two synchronized videos** (e.g., broadcast and tacticam) using deep learning and computer vision. It leverages **YOLOv8** for detection, **ByteTrack** for multi-object tracking, and custom logic for cross-camera player ID matching and visualization.

---

## Unique Features

- **Multi-Camera Player Matching:**  
  Detects and matches the same players across two different camera views, assigning consistent IDs.
- **Modular Pipeline:**  
  Each step (detection, tracking, matching, visualization) is a separate, reusable script.
- **OpenCV-based Visualization:**  
  Draws bounding boxes and player IDs on both videos for easy visual verification.
- **Customizable Matching:**  
  Uses color histograms and the Hungarian algorithm for cross-view matching, with the option to extend to deep features or tracking-based matching.

---

## Project Structure

```
cross_camera_detection/
│
├── best.pt                  # YOLOv8/YOLOv5 trained weights
├── broadcast.mp4            # Broadcast video
├── tacticam.mp4             # Tacticam video
├── detect.py                # Detection script (YOLOv8)
├── match_players.py         # Player matching script (color histogram + Hungarian)
├── visualise.py             # Visualization script
├── extract_features.py      # Feature extraction utility
├── utils.py                 # JSON utilities
├── player_id_mapping.json   # Mapping from tacticam to broadcast player IDs
├── broadcast_detections.json / tacticam_detections.json  # Detection results
├── broadcast_dets.txt / tacticam_dets.txt               # ByteTrack detection format
├── broadcast_output.mp4 / tacticam_output.mp4           # Output videos with boxes/IDs
├── ByteTrack/              # ByteTrack repo (for tracking)
└── venv/                   # (Optional) Python virtual environment
```

---

## Setup Instructions

### 1. **Clone ByteTrack and Install Requirements**

```sh
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
git submodule update --init --recursive
pip install -r requirements.txt
cd ..
pip install torch torchvision opencv-python scikit-learn
```

> **Note:**  
> Do **not** install `yolox` from PyPI. ByteTrack includes its own `yolox` code.

---

### 2. **Prepare Your Videos and Model**

- Place your `broadcast.mp4`, `tacticam.mp4`, and `best.pt` (YOLO weights) in the project root.

---

### 3. **Run Detection**

```sh
python detect.py
```
- This will generate `broadcast_detections.json`, `tacticam_detections.json`, and the corresponding `.txt` files for ByteTrack.

---

### 4. **Run ByteTrack for Tracking**

From inside the `ByteTrack` directory:

```sh
cd ByteTrack
python tools/demo_track.py --input_type video --input_path ../broadcast.mp4 --output_dir ../bytetrack_results_broadcast --save_result True --det_file ../broadcast_dets.txt

python tools/demo_track.py --input_type video --input_path ../tacticam.mp4 --output_dir ../bytetrack_results_tacticam --save_result True --det_file ../tacticam_dets.txt
cd ..
```
- This will output tracking results in the specified output directories.

---

### 5. **Match Players Across Cameras**

```sh
python match_players.py
```
- This script matches player IDs between the two videos using color histograms and saves the mapping in `player_id_mapping.json`.

---

### 6. **Visualize Results**

```sh
python visualise.py
```
- This will generate `broadcast_output.mp4` and `tacticam_output.mp4` with bounding boxes and consistent player IDs.

---

## What Could Be Improved

- **Tracking Integration:**  
  The current matching is based on color histograms and is not robust for players with similar uniforms. Integrating ByteTrack's output and matching tracks (not just detections) would improve ID consistency.
- **Deep Feature Matching:**  
  Use deep features (e.g., from a person re-identification model) for more robust cross-camera matching.
- **Temporal Consistency:**  
  Aggregate features over time for each player/track to improve matching accuracy.
- **Automatic Synchronization:**  
  Add logic to automatically synchronize frames between videos if they are not perfectly aligned.
- **Configurable Classes:**  
  Make the target class (player/ball) configurable via command-line arguments or a config file.
- **Error Handling and Logging:**  
  Add more robust error handling and logging for production use.
- **Documentation and Tests:**  
  Add docstrings, usage examples, and unit tests for each module.

---

## Limitations

- **Matching Accuracy:**  
  The current color histogram approach may fail for players with similar appearances.
- **Manual Steps:**  
  Requires manual running of each script and ByteTrack.
- **No GUI:**  
  All interaction is via command line.

---

## How to Extend

- Integrate ByteTrack's track IDs into the matching and visualization pipeline for more robust multi-frame, multi-player ID assignment.
- Use a deep learning-based feature extractor for appearance matching.
- Add a web or desktop GUI for easier use.

---
