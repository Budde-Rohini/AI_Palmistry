import os
import uuid
import cv2
import numpy as np
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from config import Config

def validate_palm_image(file_storage):
    """
    Validates an uploaded image file:
    - Checks file extension
    - Checks file size
    - Checks if image can be opened by Pillow & OpenCV
    - Checks image resolution (min 300x300)
    Returns: (is_valid: bool, message: str, cv_img: np.ndarray or None, width: int, height: int)
    """
    if not file_storage or not file_storage.filename:
        return False, "No file uploaded.", None, 0, 0

    filename = secure_filename(file_storage.filename)
    if '.' not in filename:
        return False, "Uploaded file has no extension.", None, 0, 0

    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format '.{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP.", None, 0, 0

    # Read bytes to check file size without consuming permanent stream
    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)

    if file_size > Config.MAX_CONTENT_LENGTH:
        return False, f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed limit of 10 MB.", None, 0, 0

    if file_size == 0:
        return False, "Uploaded file is empty.", None, 0, 0

    # Try opening with Pillow
    try:
        pil_img = Image.open(file_storage.stream)
        # Correct orientation based on EXIF tag if present
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img.verify()  # Verify image integrity
    except Exception as e:
        return False, "Corrupted or invalid image file. Unable to decode image.", None, 0, 0

    # Re-open stream since verify() closes/resets stream state
    file_storage.seek(0)
    try:
        pil_img = Image.open(file_storage.stream)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert to RGB mode
        rgb_img = pil_img.convert('RGB')
        width, height = rgb_img.size
    except Exception as e:
        return False, "Failed to read image pixel data.", None, 0, 0

    if width < 300 or height < 300:
        return False, f"Image resolution ({width}x{height}) is too low. Minimum 300x300 pixels required for palm analysis.", None, 0, 0

    # Convert PIL Image to OpenCV BGR numpy array
    try:
        np_img = np.array(rgb_img)
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return False, "Failed to convert image for computer vision processing.", None, 0, 0

    return True, "Image validated successfully.", cv_img, width, height


def preprocess_image(cv_img, target_width=800):
    """
    Standardizes palm image:
    - Resizes while preserving aspect ratio (target max width 800px)
    - Applies Gaussian Denoising & CLAHE contrast enhancement for line visibility.
    Returns: (preprocessed_bgr, enhanced_gray)
    """
    h, w = cv_img.shape[:2]
    if w > target_width:
        scale = target_width / float(w)
        new_w = target_width
        new_h = int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv_img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Denoise using bilateral filter to preserve line edges while smoothing skin texture
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(denoised)

    # Convert enhanced grayscale back to BGR for uniform channel handling
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return resized, enhanced_gray, enhanced_bgr

