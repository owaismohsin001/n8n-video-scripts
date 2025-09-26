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


# def overlay_translated_lines(image_path, translated_lines, font_path=None, font_size=20):
#     """
#     Draw each translated line on the image at the coordinates in translated_lines.
#     translated_lines: list of (translated_text, (x, y, w, h))
#     """
#     # Load image
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)

#     # Load font
#     font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

#     for translated_text, (x, y, w, h) in translated_lines:
#         # Cover original text area with white rectangle
#         draw.rectangle([x, y, x + w, y + h], fill="red")

#         # Draw translated text (make sure it’s a str)
#         draw.text((x, y), str(translated_text), fill="black", font=font)

#     return img


#step 2

# from PIL import Image, ImageDraw, ImageFont

# def overlay_translated_lines(image_path, translated_lines, font_path=None, font_size=20):
#     """
#     Draw each translated line on the image at the coordinates in translated_lines.
#     translated_lines: list of (translated_text, (x, y, w, h))
#     """
#     # Load image
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)

#     # Load font
#     font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

#     for translated_text, (x, y, w, h) in translated_lines:
#         # Cover original text area
#         draw.rectangle([x, y, x + w, y + h], fill="white")

#         # Get text bounding box
#         bbox = draw.textbbox((0, 0), str(translated_text), font=font)
#         text_width = bbox[2] - bbox[0]
#         text_height = bbox[3] - bbox[1]

#         # Calculate position to center text
#         text_x = x + (w - text_width) / 2
#         text_y = y + (h - text_height) / 2

#         # Draw translated text
#         draw.text((text_x, text_y), str(translated_text), fill="black", font=font)

#     return img


# step 3
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def overlay_translated_lines_on_frame(frame_bgr, translated_lines, font_path=None, font_size=20):
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
        draw.text((text_x, text_y), str(translated_text), fill="black", font=font)

    # Convert back to BGR for OpenCV
    frame_out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame_out
