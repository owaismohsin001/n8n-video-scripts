import pytesseract
from dotenv import load_dotenv
import cv2
from pytesseract import Output
import pandas as pd
from overlay_utils import overlay_translated_lines_on_frame
from translate_utils import translate_lines
from process_frame import extract_frame_from_video

load_dotenv()




reader = None

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


# step 3 latest working  using pytesseract
# def extract_lines_with_boxes_tesseract(frame_bgr, min_confidence=90, min_width=60, min_height=60, min_characters=2):
#     """
#     Accepts a BGR frame (NumPy array) directly instead of an image path.
#     Returns a list of (text, (x,y,w,h)) for each detected line.
#     """
#     # frame_bgr is already a NumPy array from cv2.VideoCapture
#     # gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     # img = gray 

#      # 1. Grayscale
#     # gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

#     # img = gray
#     img=frame_bgr
#     data = pytesseract.image_to_data(img, lang='chi_sim', output_type=Output.DICT)
#     # print(data)

#     df = pd.DataFrame(data)
#     # print(df.head(20))
#     df['conf'] = pd.to_numeric(df['conf'], errors='coerce')
#     lines = []

#     for (block_num, par_num, line_num), group in df.groupby(['block_num','par_num','line_num']):
#         # Filter words by confidence
#         group = group[group['conf'] >= min_confidence]
#         if group.empty:
#             continue

#         line_text = " ".join(word for word in group['text'] if word.strip() != "")
#         if not line_text.strip():
#             continue

#         if len(line_text.replace(" ", "")) < min_characters:
#             continue

#         x = group['left'].min()
#         y = group['top'].min()
#         w = (group['left'] + group['width']).max() - x
#         h = (group['top'] + group['height']).max() - y

#         if w >= min_width and h >= min_height:
#             lines.append((line_text, (x, y, w, h)))
#         print("Detected lines (Tesseract):", lines)
#     return lines
# frame_bgr = cv2.imread("empty_frames/frame_7100.png")
# print(extract_lines_with_boxes(frame_bgr))

import cv2
import easyocr
import re
import numpy as np

def preprocess_for_easyocr_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess a video frame for optimal EasyOCR text detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Normalize contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive threshold for variable backgrounds
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # Morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed



def clean_detected_lines(lines):
    """
    Takes a list of (text, (x, y, w, h)) and returns cleaned lines.
    Removes English, digits, and symbols — keeps only valid Chinese text.
    """
    cleaned = []
    for text, box in lines:
        # Remove everything except Chinese characters and spaces
        clean_text = re.sub(r'[^\u4e00-\u9fff\s]', '', text).strip()

        # Skip empty or very short strings
        if len(clean_text) < 2:
            continue

        cleaned.append((clean_text, box))

    return cleaned


def get_reader():
    """
    Lazy load EasyOCR reader only once.
    This prevents reloading the model on every function call.
    """
    global reader
    if reader is None:
        print("Initializing EasyOCR reader (this may take a moment on first run)...", flush=True)
        try:
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            print("EasyOCR reader loaded successfully", flush=True)
        except Exception as e:
            print(f"Error loading EasyOCR: {e}", flush=True)
            raise
    return reader

def extract_lines_with_boxes(
    frame_bgr,
    min_confidence=0.1,  # EasyOCR confidence is between 0 and 1
    min_width=20,
    min_height=20,
    min_characters=2,
    source_language="english"
):
    """
    Accepts a BGR frame (NumPy array) directly.
    Returns a list of (text, (x,y,w,h)) for each detected line using EasyOCR.
    """
    # EasyOCR expects RGB
    reader=get_reader()
    processed_frame = preprocess_for_easyocr_frame(frame_bgr)
    # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(processed_frame)  # returns list of (bbox, text, confidence)
    lines = []

    for bbox, text, conf in results:
        if conf < min_confidence:
            continue

        clean_text = text.strip()
        if len(clean_text.replace(" ", "")) < min_characters:
            continue

        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)

        if w >= min_width and h >= min_height:
            lines.append((clean_text, (x, y, w, h)))

    return lines


