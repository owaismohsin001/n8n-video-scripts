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
# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# def overlay_translated_lines_on_frame(frame_bgr, translated_lines, font_path=None, font_size=20,font_color="black"):
#     """
#     Draw each translated line on a cv2 frame (BGR).
#     translated_lines: list of (translated_text, (x, y, w, h))
#     Returns a cv2 BGR frame with overlay drawn.
#     """
#     # Convert BGR (cv2) to RGB PIL Image
#     img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
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
#         draw.text((text_x, text_y), str(translated_text), fill=font_color, font=font)

#     # Convert back to BGR for OpenCV
#     frame_out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     return frame_out
# import re
# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# from collections import Counter

# def overlay_translated_lines_on_frame(
#     frame_bgr,
#     translated_lines,
#     font_path=None,
#     font_size=20,         # kept for compatibility (ignored for auto-fit)
#     font_color="black",
#     fill_bg="white",
#     auto_detect_size=True  # NEW: Auto-detect original font size
# ):
#     """
#     Draw translated text on each (x, y, w, h) box — automatically matching
#     font size to fit the box height while preserving your old interface.

#     Args:
#         frame_bgr:         OpenCV BGR frame
#         translated_lines:  [(text, (x,y,w,h)), ...]
#         font_path:         Path to .ttf font file (same param as before)
#         font_size:         Manual font size (used if auto_detect_size=False)
#         font_color:        Text color (e.g., 'black', (r,g,b))
#         fill_bg:           Background fill inside each box ('white' or None)
#         auto_detect_size:  If True, detect font size from bounding boxes
#     """
    
#     def estimate_font_size_from_boxes(boxes, frame_height):
#         """
#         Estimate the original font size based on bounding box heights.
        
#         Strategy:
#         1. Analyze all box heights
#         2. Use median height as reference (robust to outliers)
#         3. Apply heuristic: font_size ≈ box_height * 0.7 (typical for most fonts)
#         4. Consider frame resolution for DPI estimation
#         """
#         if not boxes:
#             return 20  # fallback
        
#         heights = [h for _, (_, _, _, h) in boxes if h > 0]
#         if not heights:
#             return 20
        
#         # Use median height (more robust than mean)
#         median_height = sorted(heights)[len(heights) // 2]
        
#         # Heuristic: font point size is typically 70-75% of bounding box height
#         # This accounts for ascenders/descenders and line spacing
#         base_font_size = int(median_height * 0.72)
        
#         # Adjust based on frame resolution (DPI consideration)
#         # Standard web display: 96 DPI, HD video: ~110-120 DPI
#         # If frame is HD (1080p) or higher, slightly increase
#         if frame_height >= 1080:
#             dpi_factor = 1.1
#         elif frame_height >= 720:
#             dpi_factor = 1.0
#         else:
#             dpi_factor = 0.9
        
#         estimated_size = int(base_font_size * dpi_factor)
        
#         # Clamp to reasonable range
#         return max(12, min(estimated_size, 200))
    
#     def analyze_box_aspect_ratios(boxes):
#         """
#         Analyze aspect ratios to detect if text is single-line or multi-line.
#         Returns average character width estimation.
#         """
#         aspect_ratios = []
#         for _, (_, _, w, h) in boxes:
#             if h > 0:
#                 aspect_ratios.append(w / h)
        
#         if not aspect_ratios:
#             return None
        
#         # Average aspect ratio gives us width-to-height relationship
#         avg_ratio = sum(aspect_ratios) / len(aspect_ratios)
#         return avg_ratio
    
#     def get_text_metrics(text, font):
#         """Get accurate text metrics including ascent/descent."""
#         dummy = Image.new("RGB", (1000, 1000), (0, 0, 0))
#         draw = ImageDraw.Draw(dummy)
#         bbox = draw.textbbox((0, 0), str(text), font=font)
#         return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
#     def fit_font_to_box_with_base_size(
#         text, font_path, box_w, box_h, base_size,
#         tolerance=0.15  # Allow 15% size variation
#     ):
#         """
#         Fit font size close to base_size while ensuring it fits the box.
#         Tries to maintain consistent sizing across all text.
#         """
#         text = str(text or "")
#         if not text.strip() or box_w <= 0 or box_h <= 0:
#             return base_size
        
#         # Start with base size
#         min_size = max(8, int(base_size * (1 - tolerance)))
#         max_size = int(base_size * (1 + tolerance))
        
#         dummy = Image.new("RGB", (box_w*4+100, box_h*4+100), (0, 0, 0))
#         draw = ImageDraw.Draw(dummy)
#         target_w = int(box_w * 0.95)
#         target_h = int(box_h * 0.90)
        
#         # First, check if base_size fits
#         font = ImageFont.truetype(font_path, base_size) if font_path else ImageFont.load_default()
#         bbox = draw.textbbox((0, 0), text, font=font)
#         tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
#         if tw <= target_w and th <= target_h:
#             # Base size fits, try to go slightly larger within tolerance
#             best = base_size
#             for size in range(base_size + 1, max_size + 1):
#                 font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
#                 bbox = draw.textbbox((0, 0), text, font=font)
#                 tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#                 if tw <= target_w and th <= target_h:
#                     best = size
#                 else:
#                     break
#             return best
#         else:
#             # Base size too large, go smaller
#             for size in range(base_size - 1, min_size - 1, -1):
#                 font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
#                 bbox = draw.textbbox((0, 0), text, font=font)
#                 tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#                 if tw <= target_w and th <= target_h:
#                     return size
#             return min_size

#     # Convert frame to PIL
#     img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img)
#     frame_height = frame_bgr.shape[0]

#     # Auto-detect font size if enabled
#     if auto_detect_size:
#         base_font_size = estimate_font_size_from_boxes(translated_lines, frame_height)
#         aspect_info = analyze_box_aspect_ratios(translated_lines)
#         print(f"[Font Analysis] Detected base font size: {base_font_size}pt")
#         print(f"[Font Analysis] Frame resolution: {frame_bgr.shape[1]}x{frame_height}")
#         if aspect_info:
#             print(f"[Font Analysis] Average box aspect ratio: {aspect_info:.2f}")
#     else:
#         base_font_size = font_size

#     # Prepare all text data
#     text_data = []
#     for text, (x, y, w, h) in translated_lines:
#         x, y, w, h = int(x), int(y), int(w), int(h)
#         if w <= 0 or h <= 0:
#             continue

#         # Fit font size based on detected base size
#         if auto_detect_size:
#             fs = fit_font_to_box_with_base_size(text, font_path, w, h, base_font_size)
#         else:
#             fs = base_font_size

#         # Load font
#         font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()

#         # Measure text
#         bbox = draw.textbbox((x, y), str(text), font=font)
#         tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
#         # Calculate centered position
#         tx = x + (w - tw) / 2
#         ty = y + (h - th) / 2
        
#         # Get actual bounding box at this position
#         actual_bbox = draw.textbbox((tx, ty), str(text), font=font)
        
#         # Adjust positions to keep text within bounds
#         if actual_bbox[3] > y + h:
#             ty -= (actual_bbox[3] - (y + h))
#         if actual_bbox[1] < y:
#             ty += (y - actual_bbox[1])
        
#         # Clamp position
#         ty = max(y, min(ty, y + h - th))
#         tx = max(x, min(tx, x + w - tw))

#         text_data.append((text, x, y, w, h, tx, ty, font, fs))

#     # Log font size distribution for debugging
#     if auto_detect_size:
#         size_distribution = Counter([fs for *_, fs in text_data])
#         print(f"[Font Analysis] Size distribution: {dict(size_distribution)}")

#     # First pass: Draw all backgrounds
#     if fill_bg:
#         for text, x, y, w, h, tx, ty, font, fs in text_data:
#             draw.rectangle([x, y, x + w, y + h], fill=fill_bg)

#     # Second pass: Draw all text on top
#     for text, x, y, w, h, tx, ty, font, fs in text_data:
#         draw.text((tx, ty), str(text), fill=font_color, font=font)

#     return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)




import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

def overlay_translated_lines_on_frame(
    frame_bgr,
    translated_lines,
    font_path=None,
    font_size=20,
    font_color="black",
    fill_bg="white",
    auto_detect_size=True
):
    """
    Draw translated text on each (x, y, w, h) box with intelligent font size detection
    that accounts for different font characteristics.
    """
    
    def analyze_font_metrics(font_path, test_size=100):
        """
        Analyze the actual rendering characteristics of a font.
        Returns metrics that help us understand how this font renders.
        """
        if not font_path:
            return {
                'cap_height_ratio': 0.70,
                'x_height_ratio': 0.50,
                'baseline_ratio': 0.20,
                'width_factor': 1.0
            }
        
        try:
            font = ImageFont.truetype(font_path, test_size)
            dummy = Image.new("RGB", (2000, 500), (255, 255, 255))
            draw = ImageDraw.Draw(dummy)
            
            # Test with capital letters (cap height)
            cap_bbox = draw.textbbox((0, 0), "ABCDEFGH", font=font)
            cap_height = cap_bbox[3] - cap_bbox[1]
            
            # Test with lowercase (x-height)
            lower_bbox = draw.textbbox((0, 0), "abcdefgh", font=font)
            x_height = lower_bbox[3] - lower_bbox[1]
            
            # Test with descenders
            desc_bbox = draw.textbbox((0, 0), "gjpqy", font=font)
            full_height = desc_bbox[3] - desc_bbox[1]
            
            # Test width characteristics
            wide_bbox = draw.textbbox((0, 0), "MMMMMMMM", font=font)
            narrow_bbox = draw.textbbox((0, 0), "iiiiiiii", font=font)
            avg_width = (wide_bbox[2] - wide_bbox[0]) / 8
            
            # Calculate ratios (normalized to test_size)
            metrics = {
                'cap_height_ratio': cap_height / test_size,
                'x_height_ratio': x_height / test_size,
                'full_height_ratio': full_height / test_size,
                'baseline_ratio': (full_height - x_height) / test_size,
                'avg_char_width': avg_width / test_size,
                'actual_height': full_height  # Real pixel height at test_size
            }
            
            # print(f"[Font Analysis] Font metrics at {test_size}pt:")
            # print(f"  - Cap height ratio: {metrics['cap_height_ratio']:.3f}")
            # print(f"  - X-height ratio: {metrics['x_height_ratio']:.3f}")
            # print(f"  - Full height ratio: {metrics['full_height_ratio']:.3f}")
            # print(f"  - Actual pixel height: {metrics['actual_height']:.1f}px")
            
            return metrics
            
        except Exception as e:
            print(f"[Font Analysis] Error analyzing font: {e}")
            return {
                'cap_height_ratio': 0.70,
                'x_height_ratio': 0.50,
                'full_height_ratio': 0.75,
                'baseline_ratio': 0.20,
                'avg_char_width': 0.5
            }
    
    def estimate_font_size_from_boxes(boxes, frame_height, font_metrics):
        """
        Estimate the original font size based on bounding box heights,
        accounting for actual font rendering characteristics.
        """
        if not boxes:
            return 20
        
        heights = [h for _, (_, _, _, h) in boxes if h > 0]
        if not heights:
            return 20
        
        # Use median height
        median_height = sorted(heights)[len(heights) // 2]
        
        # Instead of fixed 0.72 factor, use actual font metrics
        # Point size needed = box_height / full_height_ratio
        # Add some padding (0.85 factor) to ensure text doesn't touch edges
        base_font_size = int((median_height * 0.85) / font_metrics['full_height_ratio'])
        
        # Adjust based on frame resolution
        if frame_height >= 1080:
            dpi_factor = 1.05
        elif frame_height >= 720:
            dpi_factor = 1.0
        else:
            dpi_factor = 0.95
        
        estimated_size = int(base_font_size * dpi_factor)
        
        # print(f"[Font Analysis] Box height: {median_height}px → Estimated font: {estimated_size}pt")
        
        return max(12, min(estimated_size, 200))
    
    def fit_font_to_box_precise(
        text, font_path, box_w, box_h, base_size, font_metrics,
        width_safety=0.95, height_safety=0.88
    ):
        """
        Precisely fit font to box using actual font metrics.
        Uses iterative refinement instead of tolerance-based approach.
        """
        text = str(text or "")
        if not text.strip() or box_w <= 0 or box_h <= 0:
            return base_size
        
        target_w = int(box_w * width_safety)
        target_h = int(box_h * height_safety)
        
        dummy = Image.new("RGB", (max(box_w*3, 1000), max(box_h*3, 500)), (255, 255, 255))
        draw = ImageDraw.Draw(dummy)
        
        # Start with base size and refine
        current_size = base_size
        
        # Quick check: does base size fit?
        font = ImageFont.truetype(font_path, current_size) if font_path else ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        if tw <= target_w and th <= target_h:
            # Try to go larger (but cautiously)
            for try_size in range(current_size + 1, current_size + 10):
                font = ImageFont.truetype(font_path, try_size) if font_path else ImageFont.load_default()
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                if tw <= target_w and th <= target_h:
                    current_size = try_size
                else:
                    break
            return current_size
        
        # Size too large, use binary search to find optimal
        lo, hi = 8, current_size - 1
        best = lo
        
        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, mid) if font_path else ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            if tw <= target_w and th <= target_h:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        return best
    
    def estimate_text_density(text):
        """
        Estimate how 'dense' the text is (wide chars vs narrow chars).
        Helps adjust width calculations.
        """
        if not text:
            return 1.0
        
        wide_chars = sum(1 for c in text if c.isupper() or c in 'MWmw@#%&')
        narrow_chars = sum(1 for c in text if c in 'iljI!|.,;:')
        total = len(text)
        
        if total == 0:
            return 1.0
        
        # Density factor: higher = wider text
        density = 1.0 + (wide_chars / total * 0.2) - (narrow_chars / total * 0.15)
        return density

    # Convert frame to PIL
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    frame_height = frame_bgr.shape[0]

    # Analyze the font first
    # print(f"[Font Analysis] Analyzing font: {font_path or 'default'}")
    font_metrics = analyze_font_metrics(font_path)

    # Auto-detect font size with font-aware analysis
    if auto_detect_size:
        base_font_size = estimate_font_size_from_boxes(
            translated_lines, frame_height, font_metrics
        )
        # print(f"[Font Analysis] Base font size: {base_font_size}pt")
        # print(f"[Font Analysis] Frame resolution: {frame_bgr.shape[1]}x{frame_height}")
    else:
        base_font_size = font_size

    # Prepare all text data
    text_data = []
    font_sizes_used = []
    
    for text, (x, y, w, h) in translated_lines:
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue

        # Adjust width safety based on text density
        text_density = estimate_text_density(text)
        width_safety = 0.95 / text_density
        
        # Fit font size precisely
        if auto_detect_size:
            fs = fit_font_to_box_precise(
                text, font_path, w, h, base_font_size, font_metrics,
                width_safety=width_safety, height_safety=0.88
            )
        else:
            fs = base_font_size
        
        font_sizes_used.append(fs)

        # Load font
        font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()

        # Measure text
        bbox = draw.textbbox((x, y), str(text), font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Calculate centered position
        tx = x + (w - tw) / 2
        ty = y + (h - th) / 2
        
        # Get actual bounding box at this position
        actual_bbox = draw.textbbox((tx, ty), str(text), font=font)
        
        # Adjust positions to keep text within bounds
        if actual_bbox[3] > y + h:
            ty -= (actual_bbox[3] - (y + h) + 2)  # +2 for extra padding
        if actual_bbox[1] < y:
            ty += (y - actual_bbox[1] + 2)
        
        # Clamp position
        ty = max(y + 2, min(ty, y + h - th - 2))
        tx = max(x + 2, min(tx, x + w - tw - 2))

        text_data.append((text, x, y, w, h, tx, ty, font, fs))

    # Log font size statistics
    if auto_detect_size and font_sizes_used:
        size_counter = Counter(font_sizes_used)
        avg_size = sum(font_sizes_used) / len(font_sizes_used)
        most_common = size_counter.most_common(1)[0]
        # print(f"[Font Analysis] Size statistics:")
        # print(f"  - Average: {avg_size:.1f}pt")
        # print(f"  - Most common: {most_common[0]}pt (used {most_common[1]} times)")
        # print(f"  - Range: {min(font_sizes_used)}-{max(font_sizes_used)}pt")

    # First pass: Draw all backgrounds
    if fill_bg:
        for text, x, y, w, h, tx, ty, font, fs in text_data:
            draw.rectangle([x, y, x + w, y + h], fill=fill_bg)

    # Second pass: Draw all text on top
    for text, x, y, w, h, tx, ty, font, fs in text_data:
        draw.text((tx, ty), str(text), fill=font_color, font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
