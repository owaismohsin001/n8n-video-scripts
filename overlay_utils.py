from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os
from openai import OpenAI
from dotenv import load_dotenv
import cv2
from pytesseract import Output
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display  

# step 3
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def overlay_translated_lines_on_frame(frame_bgr, translated_lines, font_path=None, font_size=20,font_color="black"):
    """
    Draw each translated line on a cv2 frame (BGR).
    translated_lines: list of (translated_text, (x, y, w, h))
    Returns a cv2 BGR frame with overlay drawn.
    """
    # Convert BGR (cv2) to RGB PIL Image
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # Load font
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    for translated_text, (x, y, w, h) in translated_lines:
        # Cover original text area
        draw.rectangle([x, y, x + w, y + h], fill="white")

        # Get text bounding box
        bbox = draw.textbbox((0, 0), str(translated_text), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position to center text
        text_x = x + (w - text_width) / 2
        text_y = y + (h - text_height) / 2

        # Draw translated text
        draw.text((text_x, text_y), str(translated_text), fill=font_color, font=font)

    # Convert back to BGR for OpenCV
    frame_out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame_out



# import re
# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# def overlay_translated_lines_on_frame(
#     frame_bgr,
#     translated_lines,
#     font_path=None,
#     font_size=20,         # kept for compatibility (ignored for auto-fit)
#     font_color="black",
#     fill_bg="white"
# ):
#     """
#     Draw translated text on each (x, y, w, h) box — automatically matching
#     font size to fit the box height while preserving your old interface.

#     Args:
#         frame_bgr:        OpenCV BGR frame
#         translated_lines: [(text, (x,y,w,h)), ...]
#         font_path:        Path to .ttf font file (same param as before)
#         font_size:        Ignored (kept only for backward compatibility)
#         font_color:       Text color (e.g., 'black', (r,g,b))
#         fill_bg:          Background fill inside each box ('white' or None)
#     """
#     HAN_RE = re.compile(r"[\u4e00-\u9fff]")

#     def fit_font_size_to_box(
#         text, font_path, box_w, box_h, min_size=8, max_size=300,
#         width_margin=0.98, height_margin=0.95
#     ):
#         """Binary search for max font size that fits in (w,h)."""
#         text = str(text or "")
#         if not text.strip() or box_w <= 0 or box_h <= 0:
#             return max(min_size, min(box_h, 14))

#         lo, hi, best = min_size, max_size, min_size
#         dummy = Image.new("RGB", (box_w*4+100, box_h*4+100), (0,0,0))
#         draw = ImageDraw.Draw(dummy)
#         target_w = int(box_w * width_margin)
#         target_h = int(box_h * height_margin)

#         while lo <= hi:
#             mid = (lo + hi) // 2
#             font = ImageFont.truetype(font_path, mid) if font_path else ImageFont.load_default()
#             bbox = draw.textbbox((0,0), text, font=font)
#             tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#             if tw <= target_w and th <= target_h:
#                 best = mid
#                 lo = mid + 1
#             else:
#                 hi = mid - 1
#         return best

#     # Convert frame to PIL
#     img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img)

#     for text, (x, y, w, h) in translated_lines:
#         x, y, w, h = int(x), int(y), int(w), int(h)
#         if w <= 0 or h <= 0:
#             continue

#         # Auto-compute font size
#         fs = fit_font_size_to_box(text, font_path, w, h)

#         # Load font
#         font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()

#         # Optionally clear background
#         if fill_bg:
#             draw.rectangle([x, y, x + w, y + h], fill=fill_bg)

#         # Measure and center
#         bbox = draw.textbbox((0, 0), str(text), font=font)
#         tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#         tx, ty = x + (w - tw) / 2, y + (h - th) / 2

#         # Draw text
#         draw.text((tx, ty), str(text), fill=font_color, font=font)

#     return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

