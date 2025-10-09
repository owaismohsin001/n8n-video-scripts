import pytesseract
from dotenv import load_dotenv
import cv2
from pytesseract import Output
import pandas as pd
from overlay_utils import overlay_translated_lines_on_frame
from translate_utils import translate_lines
from process_frame import extract_frame_from_video

load_dotenv()

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




# def extract_lines_with_boxes(image_path, min_confidence=70, min_width=20, min_height=20):
#     """
#     Return a list of (text, (x,y,w,h)) for each detected line, 
#     filtering by confidence and minimum box size.
#     """
#     img = cv2.imread(image_path)
#     data = pytesseract.image_to_data(img, lang='chi_sim', output_type=Output.DICT)

#     # Put into a DataFrame for easy grouping
#     df = pd.DataFrame(data)

#     # Convert confidence to numeric, ignore invalid entries
#     df['conf'] = pd.to_numeric(df['conf'], errors='coerce')
#     lines = []

#     for (block_num, par_num, line_num), group in df.groupby(['block_num','par_num','line_num']):
#         # Filter words by confidence
#         group = group[group['conf'] >= min_confidence]
#         if group.empty:
#             continue

#         # Combine all remaining words in this line
#         line_text = " ".join(word for word in group['text'] if word.strip() != "")
#         if not line_text.strip():
#             continue

#         # Compute bounding box covering the whole line
#         x = group['left'].min()
#         y = group['top'].min()
#         w = (group['left'] + group['width']).max() - x
#         h = (group['top'] + group['height']).max() - y

#         # Filter by size
#         if w >= min_width and h >= min_height:
#             lines.append((line_text, (x, y, w, h)))

#     return lines

# step 2
# def extract_lines_with_boxes(image_path, min_confidence=70, min_width=20, min_height=20, min_characters=2):
#     """
#     Return a list of (text, (x,y,w,h)) for each detected line, 
#     filtering by confidence, minimum box size, and meaningful length.
#     """
#     img = cv2.imread(image_path)
#     data = pytesseract.image_to_data(img, lang='chi_sim', output_type=Output.DICT)

#     # Put into a DataFrame for easy grouping
#     df = pd.DataFrame(data)

#     # Convert confidence to numeric, ignore invalid entries
#     df['conf'] = pd.to_numeric(df['conf'], errors='coerce')
#     lines = []

#     for (block_num, par_num, line_num), group in df.groupby(['block_num','par_num','line_num']):
#         # Filter words by confidence
#         group = group[group['conf'] >= min_confidence]
#         if group.empty:
#             continue

#         # Combine all remaining words in this line
#         line_text = " ".join(word for word in group['text'] if word.strip() != "")
#         if not line_text.strip():
#             continue

#         # Skip lines that are too short or likely meaningless
#         if len(line_text.replace(" ", "")) < min_characters:
#             continue

#         # Compute bounding box covering the whole line
#         x = group['left'].min()
#         y = group['top'].min()
#         w = (group['left'] + group['width']).max() - x
#         h = (group['top'] + group['height']).max() - y

#         # Filter by size
#         if w >= min_width and h >= min_height:
#             lines.append((line_text, (x, y, w, h)))

#     return lines


# step 3 latest working  using pytesseract
# def extract_lines_with_boxes(frame_bgr, min_confidence=10, min_width=20, min_height=20, min_characters=2):
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

#     return lines
# frame_bgr = cv2.imread("empty_frames/frame_7100.png")
# print(extract_lines_with_boxes(frame_bgr))

import cv2
import easyocr
import re

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





# Initialize the EasyOCR reader once (outside the function for performance)
reader = easyocr.Reader(['ch_sim',"en"])  # or ['en', 'chi_sim'] if multiple langs

def extract_lines_with_boxes(
    frame_bgr,
    min_confidence=0.1,  # EasyOCR confidence is between 0 and 1
    min_width=20,
    min_height=20,
    min_characters=2
):
    """
    Accepts a BGR frame (NumPy array) directly.
    Returns a list of (text, (x,y,w,h)) for each detected line using EasyOCR.
    """
    # EasyOCR expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    results = reader.readtext(frame_rgb)  # returns list of (bbox, text, confidence)
    lines = []

    for bbox, text, conf in results:
        if conf < min_confidence:
            continue

        clean_text = text.strip()
        if len(clean_text.replace(" ", "")) < min_characters:
            continue

        # bbox is 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)

        if w >= min_width and h >= min_height:
            lines.append((clean_text, (x, y, w, h)))
        print("lines before cleaning",lines)
        lines = clean_detected_lines(lines)   
        print("lines after cleaning",lines)

    return lines
# frame_bgr = cv2.imread("empty_frames/frame_7100.png")
# print(extract_lines_with_boxes(frame_bgr))


import cv2
import re
import pytesseract
from pytesseract import Output

def extract_lines_with_boxes_tesseract(
    frame_bgr,
    lang="chi_sim",           # change to 'spa', 'deu', 'chi_sim', etc. as needed
    min_confidence=40,    # pytesseract conf is 0–100
    min_width=20,
    min_height=20,
    min_characters=2,
    pad=2
):
    """
    Accepts a BGR frame (NumPy array) directly.
    Returns a list of (text, (x, y, w, h)) for each detected line using pytesseract.
    """
    # Convert to grayscale for better OCR
    # gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Simple binarization (OTSU)
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Configure tesseract: oem 3 (LSTM), psm 6 (block of text)
    # config = f"--oem 3 --psm 6 -l {lang}"

    data = pytesseract.image_to_data(frame_bgr, output_type=Output.DICT)

    lines = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        text = re.sub(r"\s+", " ", data["text"][i]).strip()
        if not text or len(text.replace(" ", "")) < min_characters:
            continue

        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1
        if conf < min_confidence:
            continue

        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        if w < min_width or h < min_height:
            continue

        # Add small padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + 2 * pad
        h = h + 2 * pad

        lines.append((text, (x, y, w, h)))

    # Sort lines top-to-bottom, left-to-right
    lines.sort(key=lambda item: (item[1][1], item[1][0]))

    return lines







#testing step 4 - save empty frames
# import cv2
# import pandas as pd
# import pytesseract
# from pytesseract import Output
# import os

# def extract_lines_with_boxes(frame_bgr,
#                              min_confidence=30,
#                              save_empty_path="empty_frames",
#                              frame_number=None):
#     """
#     Extract lines or single words from a BGR frame.
#     Saves the frame if no valid text found.
#     """

#     img = frame_bgr
#     data = pytesseract.image_to_data(img, lang='chi_sim', output_type=Output.DICT)
#     df = pd.DataFrame(data)
#     df['conf'] = pd.to_numeric(df['conf'], errors='coerce')

#     lines = []

#     # Try grouping by line
#     if 'line_num' in df.columns:
#         grouped = df.groupby(['block_num', 'par_num', 'line_num'])
#     else:
#         grouped = [(None, df)]

#     for _, group in grouped:
#         words = [w for i, w in enumerate(group['text'])
#                  if w.strip() != "" and group['conf'].iloc[i] >= min_confidence]
#         if words:
#             # bounding box covering all words
#             x = group['left'].min()
#             y = group['top'].min()
#             w = (group['left'] + group['width']).max() - x
#             h = (group['top'] + group['height']).max() - y
#             lines.append((" ".join(words), (x, y, w, h)))

#     # If still no lines, fallback to any single words above confidence
#     if not lines:
#         for i, w in enumerate(df['text']):
#             if w.strip() != "" and df['conf'].iloc[i] >= min_confidence:
#                 x = int(df['left'].iloc[i])
#                 y = int(df['top'].iloc[i])
#                 w_box = int(df['width'].iloc[i])
#                 h_box = int(df['height'].iloc[i])
#                 lines.append((w, (x, y, w_box, h_box)))

#     # Save empty frame if no lines at all
#     if not lines :
#         os.makedirs(save_empty_path, exist_ok=True)
#         filename = os.path.join(save_empty_path, f"frame_{frame_number}.png")
#         cv2.imwrite(filename, frame_bgr)
#         print(f"No lines detected — saved frame: {filename}")

#     return lines







# def extract_lines_with_boxes(image_path):
#     """Return a list of (text, (x,y,w,h)) for each detected line."""
#     img = cv2.imread(image_path)
#     data = pytesseract.image_to_data(img, lang='chi_sim', output_type=Output.DICT)

#     # Put into a DataFrame for easy grouping
#     df = pd.DataFrame(data)
#     lines = []
#     for (block_num, par_num, line_num), group in df.groupby(['block_num','par_num','line_num']):
#         # Combine all words in this line
#         line_text = " ".join(word for word in group['text'] if word.strip() != "")
#         if line_text.strip():
#             # Compute bounding box covering the whole line
#             x = group['left'].min()
#             y = group['top'].min()
#             w = (group['left'] + group['width']).max() - x
#             h = (group['top'] + group['height']).max() - y
#             lines.append((line_text, (x,y,w,h)))
#     return lines

# Example usage
# image_path = "output_images/frame_0.jpg"

# image_path=extract_frame_from_video(video_filename='test2.mp4', frame_number=7, output_dir='output_images')
# lines = extract_lines_with_boxes(image_path)
# translated_lines = translate_lines(lines, target_language="English")
# print(translated_lines)
# result_img = overlay_translated_lines(image_path, translated_lines, font_path="fonts/NotoSans-Regular.ttf", font_size=45)
# result_img.save(image_path)


# image_path = "output_images/frame_1.png"
# lines = extract_lines_with_boxes(image_path)
# translated_lines = translate_lines(lines, target_language="English")
# print(translated_lines)
# result_img = overlay_translated_lines("output_images/frame_1.png", translated_lines, font_path="fonts/NotoSans-Regular.ttf", font_size=45)
# result_img.save("output_images/frame_1_translated.png")