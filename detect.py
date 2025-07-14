# detect.py
import torch
import cv2
import os
import json
from yolov5 import YOLOv5
from ultralytics import YOLO

def detect_players(video_path, model, target_class=1):
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.3, verbose=False)
        r = results[0]

        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls == target_class:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        'frame': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls
                    })
        frame_idx += 1

    cap.release()
    return detections

def save_detections_for_bytrack(detections, output_txt):
    with open(output_txt, 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            frame = det['frame'] + 1  # ByteTrack expects 1-based frame index
            conf = det['confidence']
            class_id = det.get('class', 0)
            f.write(f"{frame},{x1},{y1},{w},{h},{conf},{class_id}\n")

if __name__ == '__main__':
    model = YOLO("best.pt")  # or "cuda" if you have a GPU

    # Broadcast
    broadcast_detections = detect_players("broadcast.mp4", model, target_class=1)
    with open("broadcast_detections.json", "w") as f:
        json.dump(broadcast_detections, f)
    save_detections_for_bytrack(broadcast_detections, "broadcast_dets.txt")

    # Tacticam
    tacticam_detections = detect_players("tacticam.mp4", model, target_class=1)
    with open("tacticam_detections.json", "w") as f:
        json.dump(tacticam_detections, f)
    save_detections_for_bytrack(tacticam_detections, "tacticam_dets.txt")

