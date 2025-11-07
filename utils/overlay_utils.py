from PIL import Image, ImageDraw, ImageFont
import cv2
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter, deque
# from collections import deque

# Keep short translation memory (thread-safe if handled per worker)
_last_translation_buffer = deque(maxlen=10)


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
            # create a dummy image of size 2000x500 with white background, on top of that we will write the text "ABCDEFGH" in black color
            dummy = Image.new("RGB", (2000, 500), (255, 255, 255))
            draw = ImageDraw.Draw(dummy)
            # here draw is the ImageDraw object, and we will use it to write the text "ABCDEFGH" in black color
            
            # Test with capital letters (cap height), to find possible required area
            cap_bbox = draw.textbbox((0, 0), "ABCDEFGH", font=font)
            cap_height = cap_bbox[3] - cap_bbox[1]
            
            # Test with lowercase (x-height), to find possible required area along x-axis
            lower_bbox = draw.textbbox((0, 0), "abcdefgh", font=font)
            x_height = lower_bbox[3] - lower_bbox[1]
            
            # Test with descenders, to find possible required area along y-axis
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
            return metrics
            
        except Exception as e:
            print(f"[Font Analysis] Error analyzing font: {e}")
            result = {
                'cap_height_ratio': 0.70,
                'x_height_ratio': 0.50,
                'full_height_ratio': 0.75,
                'baseline_ratio': 0.20,
                'avg_char_width': 0.5
            }    
            return result

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
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw <= target_w and th <= target_h:
            # Try to go larger (but cautiously)
            for try_size in range(current_size + 1, current_size + 10):
                font = ImageFont.truetype(font_path, try_size) if font_path else ImageFont.load_default()
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                
                if tw <= target_w and th <= target_h:
                    current_size = try_size
                else:
                    break
            return current_size
        
        # Size too large, use binary search to find optimal font size
        lo, hi = 8, current_size - 1
        best = lo
        
        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, mid) if font_path else ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            
            if tw <= target_w and th <= target_h:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        return best

def overlay_translated_lines_on_frame(
     frame_bgr,
    translated_lines,
    font_path=None,
    font_size=20,
    font_color="black",
    fill_bg="white",
    auto_detect_size=True,
    persistence_frames=10 
):
    """
    Draw translated text on each (x, y, w, h) box with intelligent font size detection
    that accounts for different font characteristics.
    """
    # Convert frame to PIL
        # If this frame has no text, reuse previous one if recent enough
    global _last_translation_buffer
    if not translated_lines:
        # Try to reuse last translation (only if buffer not empty)
        if _last_translation_buffer:
            translated_lines = _last_translation_buffer[-1]
        else:
            # No translation ever seen, just return frame
            return frame_bgr
    else:
        # Store this frame’s translation for continuity
        _last_translation_buffer.append(translated_lines)

    # Now proceed with your original logic
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    frame_height = frame_bgr.shape[0]

    # Analyze font
    font_metrics = analyze_font_metrics(font_path)

    # Auto-detect font size
    if auto_detect_size:
        base_font_size = estimate_font_size_from_boxes(translated_lines, frame_height, font_metrics)
    else:
        base_font_size = font_size

    text_data = []
    font_sizes_used = []

    for text, (x, y, w, h) in translated_lines:
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue

        text_density = estimate_text_density(text)
        width_safety = 0.95 / text_density

        if auto_detect_size:
            fs = fit_font_to_box_precise(
                text, font_path, w, h, base_font_size, font_metrics,
                width_safety=width_safety, height_safety=0.88
            )
        else:
            fs = base_font_size

        font_sizes_used.append(fs)
        font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()
        bbox = draw.textbbox((x, y), str(text), font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        tx = x + (w - tw) / 2
        ty = y + (h - th) / 2
        actual_bbox = draw.textbbox((tx, ty), str(text), font=font)
        if actual_bbox[3] > y + h:
            ty -= (actual_bbox[3] - (y + h) + 2)
        if actual_bbox[1] < y:
            ty += (y - actual_bbox[1] + 2)
        ty = max(y + 2, min(ty, y + h - th - 2))
        tx = max(x + 2, min(tx, x + w - tw - 2))
        text_data.append((text, x, y, w, h, tx, ty, font, fs))

    if fill_bg:
        for text, x, y, w, h, tx, ty, font, fs in text_data:
            draw.rectangle([x, y, x + w, y + h], fill=fill_bg)

    for text, x, y, w, h, tx, ty, font, fs in text_data:
        draw.text((tx, ty), str(text), fill=font_color, font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)