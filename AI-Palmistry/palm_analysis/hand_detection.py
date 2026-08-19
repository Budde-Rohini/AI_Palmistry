import os
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'hand_landmarker.task')

def ensure_model_exists():
    """Downloads MediaPipe Hand Landmarker task file if missing."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception as e:
            print(f"Warning: Failed to download MediaPipe model: {e}")

# MediaPipe 21 Landmark Indices:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

LANDMARK_NAMES = {
    0: "Wrist", 1: "Thumb_CMC", 2: "Thumb_MCP", 3: "Thumb_IP", 4: "Thumb_Tip",
    5: "Index_MCP", 6: "Index_PIP", 7: "Index_DIP", 8: "Index_Tip",
    9: "Middle_MCP", 10: "Middle_PIP", 11: "Middle_DIP", 12: "Middle_Tip",
    13: "Ring_MCP", 14: "Ring_PIP", 15: "Ring_DIP", 16: "Ring_Tip",
    17: "Pinky_MCP", 18: "Pinky_PIP", 19: "Pinky_DIP", 20: "Pinky_Tip"
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm Base Line Across Knuckles
]

def detect_hand(cv_img):
    """
    Detects hand landmarks using MediaPipe Hand Landmarker.
    If MediaPipe fails, falls back to OpenCV skin-contour detection.
    
    Returns dict containing:
    {
        'detected': bool,
        'detection_method': 'mediapipe' | 'opencv_fallback' | 'none',
        'hand_type': 'Left' | 'Right' | 'Unknown',
        'landmarks': list of (x, y) tuples,
        'bbox': (xmin, ymin, xmax, ymax),
        'palm_width': float,
        'palm_height': float,
        'aspect_ratio': float,
        'orientation_angle': float,
        'finger_lengths': dict,
        'landmark_overlay': np.ndarray BGR
    }
    """
    ensure_model_exists()
    h, w, c = cv_img.shape

    if os.path.exists(MODEL_PATH):
        try:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                num_hands=2,
                min_hand_detection_confidence=0.4
            )
            with vision.HandLandmarker.create_from_options(options) as detector:
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                )
                result = detector.detect(mp_img)

                if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                    # Select largest hand if multiple hands detected
                    best_hand_idx = 0
                    best_area = 0
                    hand_landmarks_list = []

                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        if area > best_area:
                            best_area = area
                            best_hand_idx = idx

                    lms = result.hand_landmarks[best_hand_idx]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                    # Handedness
                    hand_type = "Unknown"
                    if result.handedness and len(result.handedness) > best_hand_idx:
                        hand_type = result.handedness[best_hand_idx][0].category_name

                    # Bounding Box
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    pad_x = int((max(xs) - min(xs)) * 0.15)
                    pad_y = int((max(ys) - min(ys)) * 0.15)
                    xmin = max(0, min(xs) - pad_x)
                    ymin = max(0, min(ys) - pad_y)
                    xmax = min(w, max(xs) + pad_x)
                    ymax = min(h, max(ys) + pad_y)

                    # Measurements
                    wrist = pts[0]
                    index_mcp = pts[5]
                    middle_mcp = pts[9]
                    pinky_mcp = pts[17]

                    palm_width = math.hypot(index_mcp[0] - pinky_mcp[0], index_mcp[1] - pinky_mcp[1])
                    palm_height = math.hypot(wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1])
                    aspect_ratio = round(palm_width / palm_height, 3) if palm_height > 0 else 1.0

                    # Orientation Angle (wrist to middle MCP)
                    dx = middle_mcp[0] - wrist[0]
                    dy = middle_mcp[1] - wrist[1]
                    angle = math.degrees(math.atan2(dy, dx))

                    # Finger Lengths (MCP to Tip)
                    finger_lengths = {
                        'thumb': round(math.hypot(pts[4][0]-pts[2][0], pts[4][1]-pts[2][1]), 1),
                        'index': round(math.hypot(pts[8][0]-pts[5][0], pts[8][1]-pts[5][1]), 1),
                        'middle': round(math.hypot(pts[12][0]-pts[9][0], pts[12][1]-pts[9][1]), 1),
                        'ring': round(math.hypot(pts[16][0]-pts[13][0], pts[16][1]-pts[13][1]), 1),
                        'pinky': round(math.hypot(pts[20][0]-pts[17][0], pts[20][1]-pts[17][1]), 1)
                    }

                    # Draw Overlay
                    overlay = draw_landmarks_overlay(cv_img, pts, (xmin, ymin, xmax, ymax), hand_type)

                    return {
                        'detected': True,
                        'detection_method': 'mediapipe',
                        'hand_type': hand_type,
                        'landmarks': pts,
                        'bbox': (xmin, ymin, xmax, ymax),
                        'palm_width': round(palm_width, 1),
                        'palm_height': round(palm_height, 1),
                        'aspect_ratio': aspect_ratio,
                        'orientation_angle': round(angle, 1),
                        'finger_lengths': finger_lengths,
                        'landmark_overlay': overlay
                    }
        except Exception as e:
            print(f"MediaPipe processing error: {e}")

    # Fallback to OpenCV skin segmentation if MediaPipe fails or is unavailable
    return opencv_fallback_hand_detection(cv_img)


def opencv_fallback_hand_detection(cv_img):
    """Fallback OpenCV skin color & contour hand detection."""
    h, w, _ = cv_img.shape
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # Skin color threshold range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > (w * h * 0.05):
            x, y, bw, bh = cv2.boundingRect(largest_contour)

            # Center 60% estimated palm ROI
            pad_x = int(bw * 0.1)
            pad_y = int(bh * 0.1)
            xmin = max(0, x - pad_x)
            ymin = max(0, y - pad_y)
            xmax = min(w, x + bw + pad_x)
            ymax = min(h, y + bh + pad_y)

            overlay = cv_img.copy()
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 215, 255), 2)
            cv2.putText(overlay, "Hand Region (OpenCV Fallback)", (xmin, max(20, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

            return {
                'detected': True,
                'detection_method': 'opencv_fallback',
                'hand_type': 'Unknown',
                'landmarks': [],
                'bbox': (xmin, ymin, xmax, ymax),
                'palm_width': float(bw),
                'palm_height': float(bh),
                'aspect_ratio': round(bw / bh, 3) if bh > 0 else 1.0,
                'orientation_angle': 0.0,
                'finger_lengths': {},
                'landmark_overlay': overlay
            }

    # Center crop fallback if no clear contour found
    xmin, ymin = int(w * 0.1), int(h * 0.1)
    xmax, ymax = int(w * 0.9), int(h * 0.9)
    overlay = cv_img.copy()
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (100, 100, 100), 2)

    return {
        'detected': False,
        'detection_method': 'none',
        'hand_type': 'Unknown',
        'landmarks': [],
        'bbox': (xmin, ymin, xmax, ymax),
        'palm_width': float(xmax - xmin),
        'palm_height': float(ymax - ymin),
        'aspect_ratio': 1.0,
        'orientation_angle': 0.0,
        'finger_lengths': {},
        'landmark_overlay': overlay
    }


def draw_landmarks_overlay(cv_img, landmarks, bbox, hand_type):
    """Draws sleek gold/purple landmark lines and nodes over the hand image."""
    overlay = cv_img.copy()

    # Draw connection lines (purple glow)
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            cv2.line(overlay, pt1, pt2, (179, 82, 121), 2, cv2.LINE_AA)

    # Draw landmark nodes (gold circles)
    for idx, (x, y) in enumerate(landmarks):
        color = (47, 186, 243) if idx in [0, 5, 9, 13, 17] else (255, 255, 255)
        radius = 5 if idx in [0, 5, 9, 13, 17] else 3
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)

    # Draw bounding box
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (47, 186, 243), 2)
    cv2.putText(overlay, f"Hand: {hand_type} (MediaPipe)", (xmin, max(20, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (47, 186, 243), 2, cv2.LINE_AA)

    return overlay
