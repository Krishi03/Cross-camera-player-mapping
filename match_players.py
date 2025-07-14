import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_color_histogram(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((256,))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def group_detections_by_frame(detections):
    frames = defaultdict(list)
    for det in detections:
        frames[det["frame"]].append(det)
    return frames

def match_players(broadcast_json, tacticam_json, broadcast_video, tacticam_video, max_frames=50):
    broadcast_dets = load_json(broadcast_json)
    tacticam_dets = load_json(tacticam_json)

    broadcast_frames = group_detections_by_frame(broadcast_dets)
    tacticam_frames = group_detections_by_frame(tacticam_dets)

    match_counts = defaultdict(Counter)

    # Open videos for frame extraction
    cap_broadcast = cv2.VideoCapture(broadcast_video)
    cap_tacticam = cv2.VideoCapture(tacticam_video)

    shared_frames = sorted(set(broadcast_frames) & set(tacticam_frames))
    frames_used = 0

    for frame in shared_frames:
        if frames_used >= max_frames:
            break

        # Read frames from both videos
        cap_broadcast.set(cv2.CAP_PROP_POS_FRAMES, frame)
        cap_tacticam.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret_b, img_b = cap_broadcast.read()
        ret_t, img_t = cap_tacticam.read()
        if not (ret_b and ret_t):
            continue

        broadcast_histograms = [extract_color_histogram(img_b, det["bbox"]) for det in broadcast_frames[frame]]
        tacticam_histograms = [extract_color_histogram(img_t, det["bbox"]) for det in tacticam_frames[frame]]

        if not broadcast_histograms or not tacticam_histograms:
            continue

        sim_matrix = cosine_similarity(tacticam_histograms, broadcast_histograms)
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # maximize similarity

        for tidx, bidx in zip(row_ind, col_ind):
            match_counts[tidx][bidx] += 1

        frames_used += 1

    cap_broadcast.release()
    cap_tacticam.release()

    # Assign the most frequent match for each tacticam player
    id_map = {str(tidx): int(counts.most_common(1)[0][0]) for tidx, counts in match_counts.items() if counts}

    with open('player_id_mapping.json', 'w') as f:
        json.dump(id_map, f)

    print("Tacticam to Broadcast Player ID Mapping:")
    print(id_map)

if __name__ == '__main__':
    match_players(
        "broadcast_detections.json",
        "tacticam_detections.json",
        "broadcast.mp4",
        "tacticam.mp4",
        max_frames=50  # You can increase for more robust matching
    )
