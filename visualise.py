import json
import cv2

def load_detections(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def draw_detections(video_path, detections, output_path, id_mapping=None, source='broadcast'):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (
        int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = [d for d in detections if d['frame'] == frame_idx]

        for idx, det in enumerate(frame_dets):
            # If you're using class info in detections (optional)
            if 'class' in det and det['class'] != 0:  # ⚠️ Skip non-players
                continue

            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0)
            thickness = 2

            # Match ID
            if source == 'tacticam' and id_mapping:
                display_id = id_mapping.get(str(idx), idx)
            else:
                display_id = idx

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f'ID {display_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[✓] Output video saved to {output_path}")

if __name__ == "__main__":
    broadcast_dets = load_detections("broadcast_detections.json")
    tacticam_dets = load_detections("tacticam_detections.json")
    with open("player_id_mapping.json", 'r') as f:
        id_mapping = json.load(f)

    draw_detections("broadcast.mp4", broadcast_dets, "broadcast_output.mp4", source='broadcast')
    draw_detections("tacticam.mp4", tacticam_dets, "tacticam_output.mp4", id_mapping, source='tacticam')
