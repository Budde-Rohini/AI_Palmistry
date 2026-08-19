import math
import cv2
import numpy as np

def extract_palm_roi(cv_img, hand_data):
    """
    Extracts the palm Region of Interest (ROI) from hand detection landmark data.
    Aligns hand orientation so palm points straight upright.
    
    Returns:
        palm_roi: Cropped & aligned BGR image of palm
        roi_bbox: (xmin, ymin, xmax, ymax) in relative/original image coordinates
    """
    h, w, c = cv_img.shape

    if not hand_data or not hand_data.get('detected'):
        # Center 70% ROI fallback
        xmin, ymin = int(w * 0.15), int(h * 0.15)
        xmax, ymax = int(w * 0.85), int(h * 0.85)
        return cv_img[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)

    landmarks = hand_data.get('landmarks', [])

    if len(landmarks) >= 21:
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        pinky_mcp = landmarks[17]

        # Calculate rotation angle to align wrist -> middle_mcp vertically (upward)
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Desired upright angle is -90 degrees (270 deg)
        rotation_angle = angle_deg - (-90)

        # Rotate original image around wrist coordinate
        M = cv2.getRotationMatrix2D((wrist[0], wrist[1]), rotation_angle, 1.0)
        rotated_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Transform landmarks to rotated space to compute aligned ROI bounding box
        pts_np = np.array(landmarks, dtype=np.float32)
        ones = np.ones((pts_np.shape[0], 1), dtype=np.float32)
        pts_homo = np.hstack([pts_np, ones])
        rotated_pts = M.dot(pts_homo.T).T

        # Palm boundary landmarks (wrist, index_mcp, pinky_mcp, thumb_cmc)
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        palm_pts = rotated_pts[palm_indices]

        xs = palm_pts[:, 0]
        ys = palm_pts[:, 1]

        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))

        # Add 10% padding
        bw = max_x - min_x
        bh = max_y - min_y
        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.12)

        crop_xmin = max(0, int(min_x) - pad_x)
        crop_ymin = max(0, int(min_y) - pad_y)
        crop_xmax = min(w, int(max_x) + pad_x)
        crop_ymax = min(h, int(max_y) + pad_y)

        # Ensure valid non-zero crop area
        if crop_xmax > crop_xmin + 50 and crop_ymax > crop_ymin + 50:
            palm_roi = rotated_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            return palm_roi, (crop_xmin, crop_ymin, crop_xmax, crop_ymax)

    # Bounding box fallback
    xmin, ymin, xmax, ymax = hand_data.get('bbox', (0, 0, w, h))
    palm_roi = cv_img[ymin:ymax, xmin:xmax]
    if palm_roi.size == 0:
        palm_roi = cv_img.copy()
        xmin, ymin, xmax, ymax = 0, 0, w, h
    return palm_roi, (xmin, ymin, xmax, ymax)
