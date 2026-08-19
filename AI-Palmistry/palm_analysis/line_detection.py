import cv2
import numpy as np
import math

def detect_palm_lines(palm_roi, hand_type="Unknown"):
    """
    Analyzes palm ROI using OpenCV image processing to detect major creases:
    - Heart Line
    - Head Line
    - Life Line
    - Fate Line

    Returns dict with line metrics and color-coded overlay image.
    """
    if palm_roi is None or palm_roi.size == 0:
        return get_empty_line_result(palm_roi)

    h, w = palm_roi.shape[:2]

    # Preprocessing for crease extraction
    gray = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2GRAY) if len(palm_roi.shape) == 3 else palm_roi.copy()

    # Apply CLAHE to boost crease contrast against skin
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Black Top-Hat morphological operation to isolate dark lines (creases) on lighter skin background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    # Bilateral smoothing to remove noise while keeping sharp edges
    smoothed = cv2.bilateralFilter(tophat, d=7, sigmaColor=50, sigmaSpace=50)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )

    # Remove small speckles using morphological opening
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)

    # Find crease contours
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter out noise contours smaller than 25px
    valid_contours = [c for c in contours if cv2.arcLength(c, False) > 25]

    # Analyze and classify major lines
    heart_line = classify_heart_line(valid_contours, w, h, hand_type)
    head_line = classify_head_line(valid_contours, w, h, hand_type)
    life_line = classify_life_line(valid_contours, w, h, hand_type)
    fate_line = classify_fate_line(valid_contours, w, h, hand_type)

    lines_data = {
        'heart_line': heart_line,
        'head_line': head_line,
        'life_line': life_line,
        'fate_line': fate_line
    }

    # Render line overlay image
    overlay_img = draw_line_overlay(palm_roi, lines_data)

    return {
        'lines': lines_data,
        'processed_roi': enhanced,
        'overlay_image': overlay_img
    }


def classify_heart_line(contours, w, h, hand_type):
    """
    Heart Line: Upper palm region (0.15h to 0.45h).
    Runs horizontally across the upper palm below knuckles.
    """
    best_contour = None
    best_score = -1

    for c in contours:
        pts = c.reshape(-1, 2)
        ys = pts[:, 1]
        xs = pts[:, 0]

        mean_y = np.mean(ys)
        span_x = np.max(xs) - np.min(xs)
        arc_len = cv2.arcLength(c, False)

        # Heart line should be in upper 15% to 45% of palm height
        if 0.12 * h <= mean_y <= 0.45 * h and span_x > 0.15 * w:
            score = arc_len * (span_x / w)
            if score > best_score:
                best_score = score
                best_contour = pts

    if best_contour is not None and len(best_contour) > 5:
        return compute_line_metrics(best_contour, w, h, "Heart Line")
    
    # Fallback heuristic line estimation if explicit contour is faint
    y_heart = int(0.28 * h)
    pts = np.array([[int(0.2 * w), y_heart], [int(0.5 * w), int(y_heart - 0.05 * h)], [int(0.8 * w), y_heart]], dtype=np.int32)
    metrics = compute_line_metrics(pts, w, h, "Heart Line")
    metrics['detected'] = False
    metrics['note'] = "Line faint or partially visible"
    return metrics


def classify_head_line(contours, w, h, hand_type):
    """
    Head Line: Middle palm region (0.35h to 0.65h).
    Runs diagonally across center of palm.
    """
    best_contour = None
    best_score = -1

    for c in contours:
        pts = c.reshape(-1, 2)
        ys = pts[:, 1]
        xs = pts[:, 0]

        mean_y = np.mean(ys)
        span_x = np.max(xs) - np.min(xs)
        arc_len = cv2.arcLength(c, False)

        if 0.35 * h <= mean_y <= 0.65 * h and span_x > 0.18 * w:
            score = arc_len * (span_x / w)
            if score > best_score:
                best_score = score
                best_contour = pts

    if best_contour is not None and len(best_contour) > 5:
        return compute_line_metrics(best_contour, w, h, "Head Line")

    y_head = int(0.48 * h)
    pts = np.array([[int(0.18 * w), y_head], [int(0.5 * w), int(y_head + 0.03 * h)], [int(0.78 * w), int(y_head + 0.08 * h)]], dtype=np.int32)
    metrics = compute_line_metrics(pts, w, h, "Head Line")
    metrics['detected'] = False
    metrics['note'] = "Line faint or partially visible"
    return metrics


def classify_life_line(contours, w, h, hand_type):
    """
    Life Line: Curves around Thenar mount (thumb base).
    Vertical/curved line on left or right lower side.
    """
    best_contour = None
    best_score = -1

    for c in contours:
        pts = c.reshape(-1, 2)
        ys = pts[:, 1]
        xs = pts[:, 0]

        span_y = np.max(ys) - np.min(ys)
        arc_len = cv2.arcLength(c, False)

        # Life line spans vertically through middle and lower palm
        if np.max(ys) > 0.5 * h and span_y > 0.25 * h:
            score = arc_len * (span_y / h)
            if score > best_score:
                best_score = score
                best_contour = pts

    if best_contour is not None and len(best_contour) > 5:
        return compute_line_metrics(best_contour, w, h, "Life Line")

    pts = np.array([[int(0.22 * w), int(0.4 * h)], [int(0.35 * w), int(0.6 * h)], [int(0.3 * w), int(0.85 * h)]], dtype=np.int32)
    metrics = compute_line_metrics(pts, w, h, "Life Line")
    metrics['detected'] = False
    metrics['note'] = "Line faint or partially visible"
    return metrics


def classify_fate_line(contours, w, h, hand_type):
    """
    Fate Line: Vertical line running up central axis of palm (0.4w to 0.6w).
    """
    best_contour = None
    best_score = -1

    for c in contours:
        pts = c.reshape(-1, 2)
        ys = pts[:, 1]
        xs = pts[:, 0]

        mean_x = np.mean(xs)
        span_y = np.max(ys) - np.min(ys)
        arc_len = cv2.arcLength(c, False)

        if 0.3 * w <= mean_x <= 0.7 * w and span_y > 0.2 * h:
            score = arc_len * (span_y / h)
            if score > best_score:
                best_score = score
                best_contour = pts

    if best_contour is not None and len(best_contour) > 5:
        return compute_line_metrics(best_contour, w, h, "Fate Line")

    pts = np.array([[int(0.5 * w), int(0.85 * h)], [int(0.5 * w), int(0.5 * h)], [int(0.5 * w), int(0.3 * h)]], dtype=np.int32)
    metrics = compute_line_metrics(pts, w, h, "Fate Line")
    metrics['detected'] = False
    metrics['note'] = "Line faint or non-prominent"
    return metrics


def compute_line_metrics(pts, w, h, line_name):
    """Computes arc length, curvature, continuity, and depth for a line contour."""
    if pts is None or len(pts) == 0:
        return {'detected': False, 'length': 0, 'curvature': 'unknown', 'depth': 'shallow', 'continuity': 'fragmented', 'points': []}

    # Sort points along main axis
    pts_list = pts.tolist()
    arc_len = 0.0
    for i in range(len(pts) - 1):
        arc_len += math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])

    # Direct start-to-end distance (chord)
    start_pt = pts[0]
    end_pt = pts[-1]
    chord = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])

    # Curvature ratio = Arc Length / Chord Length
    curve_ratio = (arc_len / chord) if chord > 0 else 1.0
    if curve_ratio > 1.25:
        curvature = "Strong / Curved"
    elif curve_ratio > 1.08:
        curvature = "Moderate"
    else:
        curvature = "Straight / Linear"

    # Length category relative to palm dimension
    norm_len = arc_len / float(max(w, h))
    if norm_len > 0.5:
        length_cat = "Long"
    elif norm_len > 0.25:
        length_cat = "Medium"
    else:
        length_cat = "Short"

    return {
        'detected': True,
        'line_name': line_name,
        'length_px': round(arc_len, 1),
        'length': length_cat,
        'curvature_ratio': round(curve_ratio, 2),
        'curvature': curvature,
        'depth': 'Clear' if arc_len > 80 else 'Moderate',
        'continuity': 'Continuous' if len(pts) > 30 else 'Fragmented',
        'points': pts_list
    }


def draw_line_overlay(palm_roi, lines_data):
    """Draws color-coded line overlays with labels on the palm image."""
    overlay = palm_roi.copy()

    # Color Scheme (BGR)
    colors = {
        'heart_line': (180, 105, 255),  # Crimson / Pink
        'head_line': (255, 215, 0),    # Cyan / Gold-Blue
        'life_line': (50, 205, 50),     # Emerald Green
        'fate_line': (0, 215, 255)      # Bright Yellow
    }

    labels = {
        'heart_line': 'Heart Line',
        'head_line': 'Head Line',
        'life_line': 'Life Line',
        'fate_line': 'Fate Line'
    }

    for key, data in lines_data.items():
        pts = data.get('points', [])
        color = colors.get(key, (0, 255, 255))
        label = labels.get(key, key)
        is_detected = data.get('detected', False)

        if pts and len(pts) > 1:
            pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            thickness = 3 if is_detected else 2
            line_type = cv2.LINE_AA

            # Draw line curve
            cv2.polylines(overlay, [pts_array], isClosed=False, color=color, thickness=thickness, lineType=line_type)

            # Draw label tag at start of line
            start_x, start_y = pts[0]
            tag_text = f"{label}" if is_detected else f"{label} (Faint)"
            cv2.putText(overlay, tag_text, (start_x + 5, max(15, start_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return overlay


def get_empty_line_result(palm_roi):
    empty_line = {'detected': False, 'length': 'Short', 'curvature': 'Linear', 'depth': 'Shallow', 'continuity': 'None', 'points': []}
    return {
        'lines': {
            'heart_line': dict(empty_line, line_name='Heart Line'),
            'head_line': dict(empty_line, line_name='Head Line'),
            'life_line': dict(empty_line, line_name='Life Line'),
            'fate_line': dict(empty_line, line_name='Fate Line')
        },
        'processed_roi': palm_roi if palm_roi is not None else np.zeros((100,100), dtype=np.uint8),
        'overlay_image': palm_roi if palm_roi is not None else np.zeros((100,100,3), dtype=np.uint8)
    }
