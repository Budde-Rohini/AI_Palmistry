def extract_features(hand_data, line_data):
    """
    Synthesizes computer vision hand detection metrics and line analysis data
    into a structured feature dictionary.

    Guarantees no random values: uses strictly extracted measurements or
    explicitly marks unavailable metrics.
    """
    if not hand_data:
        hand_data = {}
    if not line_data or 'lines' not in line_data:
        lines = {}
    else:
        lines = line_data['lines']

    # Hand & Palm Dimensions
    hand_type = hand_data.get('hand_type', 'Unknown')
    palm_width = hand_data.get('palm_width', 0.0)
    palm_height = hand_data.get('palm_height', 0.0)
    aspect_ratio = hand_data.get('aspect_ratio', 1.0)
    finger_lengths = hand_data.get('finger_lengths', {})

    # Determine Palm Shape Element Classification based on aspect ratio & finger lengths
    # Traditional Palmistry Shape Categories:
    # Earth: Square palm, short fingers
    # Air: Square palm, long fingers
    # Fire: Long palm, short fingers
    # Water: Long palm, long fingers
    middle_finger = finger_lengths.get('middle', 0.0)
    if aspect_ratio >= 0.95:
        palm_shape = "Earth (Square)" if middle_finger < palm_height * 0.8 else "Air (Square / Long Fingers)"
    else:
        palm_shape = "Fire (Rectangular / Short Fingers)" if middle_finger < palm_height * 0.8 else "Water (Rectangular / Long Fingers)"

    def format_line(line_dict):
        if not line_dict:
            return {
                "detected": False,
                "length": "Unavailable",
                "curvature": "Unavailable",
                "depth": "Unavailable",
                "continuity": "Unavailable"
            }
        is_detected = line_dict.get('detected', False)
        return {
            "detected": is_detected,
            "length": line_dict.get('length', 'Unavailable') if is_detected else "Unavailable",
            "length_px": line_dict.get('length_px', 0.0) if is_detected else 0.0,
            "curvature": line_dict.get('curvature', 'Unavailable') if is_detected else "Unavailable",
            "depth": line_dict.get('depth', 'Unavailable') if is_detected else "Unavailable",
            "continuity": line_dict.get('continuity', 'Unavailable') if is_detected else "Unavailable"
        }

    life_line = format_line(lines.get('life_line'))
    head_line = format_line(lines.get('head_line'))
    heart_line = format_line(lines.get('heart_line'))
    fate_line = format_line(lines.get('fate_line'))

    feature_vector = {
        "hand": hand_type,
        "detection_method": hand_data.get('detection_method', 'none'),
        "palm": {
            "width_px": palm_width,
            "height_px": palm_height,
            "aspect_ratio": aspect_ratio,
            "element_shape": palm_shape,
            "finger_lengths": finger_lengths
        },
        "life_line": life_line,
        "head_line": head_line,
        "heart_line": heart_line,
        "fate_line": fate_line,
        "quality_score": calculate_quality_score(hand_data, lines)
    }

    return feature_vector


def calculate_quality_score(hand_data, lines):
    """Calculates image analysis quality score (0 to 100) based on detection confidence."""
    score = 40  # Base image baseline

    if hand_data.get('detected', False):
        if hand_data.get('detection_method') == 'mediapipe':
            score += 30
        else:
            score += 15

    detected_lines_count = sum(1 for line in lines.values() if isinstance(line, dict) and line.get('detected', False))
    score += (detected_lines_count * 7.5)

    return min(100, int(score))
