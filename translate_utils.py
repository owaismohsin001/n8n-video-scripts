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
def translate_text(text, target_language="English"):
    """Translate a single string into the target language using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o",    # or gpt-4o-mini if you prefer faster/cheaper
        # messages=[
        #     # {"role": "system", "content": "You are a translator."},
		# 	# {"role": "system", "content": "You are a translator. Only return the translated text, no explanations or extra words."},
        #     {"role": "system", "content": "You are a professional translator. Translate text exactly and accurately into the target language. Do not add any explanations, comments, or extra words. Include numbers, punctuation, symbols, and single characters as they appear in the source text. Return only the translated text, nothing else."},
        #     {"role": "user", "content": f"Translate the following text into {target_language}:\n{text}"}
        # ]
        messages = [
            {
            "role": "system",
            "content": (
                "You are a professional subtitle translator. "
                "Your job is to accurately translate text into the target language. "
                "You must detect the source language automatically and translate it completely. "
                "Follow these rules strictly:\n"
                "- Output must be only in the target language.\n"
                "- Do NOT mix with or keep any words or characters from the original language.\n"
                "- Keep numbers, punctuation, and symbols exactly as in the original.\n"
                "- Keep the meaning and tone natural for a native speaker of the target language.\n"
                "- Do NOT add explanations, notes, or formatting.\n"
                "- Do NOT output the original text, labels, or extra commentary.\n"
                "- Return only the translated text, preserving line breaks if multiple lines are given."
            ),
            },
            {
                "role": "user",
                "content": f"Translate the following text into {target_language}:\n{text.strip()}"
            }
        ]

    )
    translated_text = response.choices[0].message.content.strip()
    return translated_text


# 2. Whole-lines translator
def translate_lines(lines, target_language="English"):
    print("target_language",target_language)
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
