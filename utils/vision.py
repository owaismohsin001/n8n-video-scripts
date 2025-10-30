import cv2
import numpy as np

def preprocess_for_ocr(frame_bgr):
    """
    Preprocess BGR frame for better OCR detection.
    This version:
    1. Converts to grayscale.
    2. Uses CLAHE on grayscale (before threshold).
    3. Skips threshold unless absolutely needed.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Optional: slight denoise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)  # very small blur

    # 3. Enhance contrast BEFORE threshold
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 4. Don’t threshold yet; Tesseract works best on this
    return enhanced




def preprocess_frame_for_better_precision(frame: np.ndarray) -> np.ndarray:
    """Preprocess a video frame for optimal EasyOCR text detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Normalize contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def get_frame_at_index(video_path, frame_index):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        cap.release()
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_index < 0 or frame_index >= total_frames:
        print(f"Error: Frame index {frame_index} out of range (0-{total_frames-1})")
        cap.release()
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_index}")
        return None
    return frame
