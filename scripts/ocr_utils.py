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


# Load .env variables into environment
load_dotenv()

import os
from openai import OpenAI

# 1. Single line translator
def translate_text(text, target_language="Urdu"):
    """Translate a single string into the target language using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o",    # or gpt-4o-mini if you prefer faster/cheaper
        messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": f"Translate this text into {target_language} and also decide one meaning if two or more exost for a word: {text}"}
        ]
    )
    translated_text = response.choices[0].message.content.strip()
    return translated_text


# 2. Whole-lines translator
def translate_lines(lines, target_language="Urdu"):
    """
    Translate a list of (text, box) tuples and return
    a list of (translated_text, box) tuples.
    """
    translated_lines = []
    for text, box in lines:
        translated_text = translate_text(text, target_language)
        # convert np.int64 to int for safety
        x, y, w, h = [int(v) for v in box]
        translated_lines.append((translated_text, (x, y, w, h)))
    return translated_lines




def overlay_translated_lines(image_path, translated_lines, font_path=None, font_size=20):
    """
    Draw each translated line on the image at the coordinates in translated_lines.
    translated_lines: list of (translated_text, (x, y, w, h))
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Load font
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    for translated_text, (x, y, w, h) in translated_lines:
        # Cover original text area with white rectangle
        draw.rectangle([x, y, x + w, y + h], fill="white")

        # Draw translated text (make sure it’s a str)
        draw.text((x, y), str(translated_text), fill="black", font=font)

    return img


def extract_lines_with_boxes(image_path):
    """Return a list of (text, (x,y,w,h)) for each detected line."""
    img = cv2.imread(image_path)
    data = pytesseract.image_to_data(img,lang="chi_tra", output_type=Output.DICT)

    # Put into a DataFrame for easy grouping
    df = pd.DataFrame(data)
    lines = []
    for (block_num, par_num, line_num), group in df.groupby(['block_num','par_num','line_num']):
        # Combine all words in this line
        line_text = " ".join(word for word in group['text'] if word.strip() != "")
        if line_text.strip():
            # Compute bounding box covering the whole line
            x = group['left'].min()
            y = group['top'].min()
            w = (group['left'] + group['width']).max() - x
            h = (group['top'] + group['height']).max() - y
            lines.append((line_text, (x,y,w,h)))
    return lines

# Example usage
image_path = "output_images/frame_0.jpg"
lines = extract_lines_with_boxes(image_path)
translated_lines = translate_lines(lines, target_language="French")
print(translated_lines)

result_img = overlay_translated_lines("output_images/frame_0.jpg", translated_lines, font_path="fonts/PlaywriteFRModerne-Regular.ttf", font_size=20)
result_img.save("output_images/frame_0_translated.jpg")