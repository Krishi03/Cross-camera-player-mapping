# extract_features.py
import cv2
import numpy as np

def extract_color_histogram(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256]*3).flatten()
    hist /= (hist.sum() + 1e-6)
    return hist
