
from dotenv import load_dotenv

from utils.overlay_utils import overlay_translated_lines_on_frame
from utils.translate_utils import translate_lines
from utils.process_frame import extract_frame_from_video
import easyocr
import re
from constants.ocr import HAN_RE, LATIN_LETTER_RE, VOWEL_RE, ALNUM_OR_HAN_RE, SYMBOL_ONLY_RE, PUNCT_SIMPLE_RE
import re
from dataclasses import dataclass
from typing import List, Tuple
from utils.vision import preprocess_frame_for_better_precision, preprocess_for_ocr

load_dotenv()

reader = None



# ------------- Script / char classes -------------

# ------------- Config -------------
@dataclass
class CleanConfig:
    min_total_len: int = 2                # drop super short lines
    min_alnum_or_han: int = 2             # need at least 2 "meaningful" chars
    max_symbol_ratio: float = 0.5         # if >50% are symbols/punct, drop
    max_mixed_scripts_tiny: bool = True   # drop lines that mix scripts but are tiny
    keep_numeric_units: bool = True       # keep things like "50%", "x2", "3.0", etc.
    min_han_ratio_for_cjk: float = 0.6    # if line has any Han, require it's mostly Han
    allow_single_word_caps: bool = False  # drop single ALLCAP short shards like "MA"
    min_latin_word_len: int = 2           # require at least one latin word of length >= 2
    require_vowel_for_latin: bool = True  # latin lines need at least one vowel (English-ish)
    debug: bool = False

def _ratio(n, d):
    return 0 if d == 0 else (n / d)

def _is_symbol_only(s: str) -> bool:
    return bool(SYMBOL_ONLY_RE.match(s))

def _punct_ratio(s: str) -> float:
    # Try broad punct class; fall back to simple
    # (If you install `regex` library, replace with it for better Unicode punctuation coverage)
    punct = sum(1 for ch in s if PUNCT_SIMPLE_RE.match(ch))
    return _ratio(punct, len(s))

def _has_units_like_number(s: str) -> bool:
    # "50%", "3.0", "x2", "$5", "5k", etc.
    return bool(re.search(r"(\d[\d.,]*\s*[%%kKxX]|[\$€£]\s*\d|[xX]\s*\d)", s))

def _script_counts(s: str):
    han = sum(1 for ch in s if HAN_RE.match(ch))
    latin = sum(1 for ch in s if LATIN_LETTER_RE.match(ch))
    digits = sum(1 for ch in s if ch.isdigit())
    alnum_or_han = sum(1 for ch in s if ALNUM_OR_HAN_RE.match(ch))
    return han, latin, digits, alnum_or_han

def _looks_like_legit_latin(s: str, cfg: CleanConfig) -> bool:
    # Needs at least one latin word length >= min_latin_word_len
    words = re.findall(r"[A-Za-z]+", s)
    if not any(len(w) >= cfg.min_latin_word_len for w in words):
        return False
    if cfg.require_vowel_for_latin and not VOWEL_RE.search(s):
        return False
    return True

def _looks_like_legit_cjk(s: str, cfg: CleanConfig) -> bool:
    # For Chinese-like lines, require that majority are Han chars
    han, latin, digits, alnum_or_han = _script_counts(s)
    if han == 0:
        return False
    # If there is Han, require it's the majority of meaningful chars
    return _ratio(han, max(1, alnum_or_han)) >= cfg.min_han_ratio_for_cjk

def _is_mixed_scripts_weird_and_tiny(s: str) -> bool:
    han, latin, digits, alnum_or_han = _script_counts(s)
    # "weird tiny mix": both han and latin present but very few letters overall
    if han > 0 and latin > 0 and alnum_or_han <= 4:
        return True
    return False

def _is_all_caps_short_shard(s: str) -> bool:
    # E.g., "MA.", "OK", "X.", "LM" -> often OCR shards if very short
    letters = re.findall(r"[A-Za-z]", s)
    if 1 <= len(letters) <= 3 and "".join(letters).isupper():
        return True
    return False

def _should_keep(text: str, cfg: CleanConfig) -> bool:
    s = text.strip()
    if len(s) < cfg.min_total_len:
        return False
    if _is_symbol_only(s):
        return False

    han, latin, digits, alnum_or_han = _script_counts(s)
    # If almost nothing meaningful
    if alnum_or_han < cfg.min_alnum_or_han and not (cfg.keep_numeric_units and _has_units_like_number(s)):
        return False

    # Excessive punctuation / symbol ratio
    if _punct_ratio(s) > cfg.max_symbol_ratio:
        # unless it's a numeric-unit pattern like "50%" which is useful
        if not (cfg.keep_numeric_units and _has_units_like_number(s)):
            return False

    # Mixed scripts, tiny content → likely garbage
    if cfg.max_mixed_scripts_tiny and _is_mixed_scripts_weird_and_tiny(s):
        return False

    # Heuristics per script
    # If line contains Han anywhere, treat it as CJK; otherwise treat as Latin/other
    if han > 0:
        if not _looks_like_legit_cjk(s, cfg):
            return False
    else:
        # Latin-ish
        if latin > 0:
            if not _looks_like_legit_latin(s, cfg):
                return False
            if not cfg.allow_single_word_caps and _is_all_caps_short_shard(s):
                return False
        else:
            # No Han, no Latin letters → keep only if numeric units (e.g., "50%")
            if not (cfg.keep_numeric_units and _has_units_like_number(s)):
                # e.g., random symbols with a digit sprinkled in → drop
                return False

    return True

def clean_extracted_lines(
    lines: List[Tuple[str, Tuple[int, int, int, int]]],
    config: CleanConfig = CleanConfig()
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Filters out garbage OCR lines (symbol-only, high-punctuation, tiny mixed-script shards,
    random Chinese fragments, short ALLCAP shards like '+MA.', etc.) while preserving
    legit English/Chinese lines and useful numeric tokens like '50%'.

    Args:
        lines: [(text, (x, y, w, h)), ...]
        config: CleanConfig thresholds

    Returns:
        Filtered list of (text, box)
    """
    kept = []
    if config.debug:
        dropped = []

    for text, box in lines:
        if _should_keep(text, config):
            kept.append((text.strip(), tuple(int(v) for v in box)))
        elif config.debug:
            dropped.append((text, box))

    if config.debug:
        print("----- CLEANER DEBUG -----")
        print(f"Kept: {len(kept)}   Dropped: {len(dropped)}")
        for t, b in dropped[:20]:
            print(f"DROP: '{t}'  @ {b}")
        print("-------------------------")

    return kept

# ---------------- Example usage ----------------
# raw_lines = extract_lines_with_boxes(frame)  # -> [(text, (x,y,w,h)), ...]
# filtered = clean_extracted_lines(raw_lines, CleanConfig(debug=True))



def get_reader():
    """
    Lazy load EasyOCR reader only once.
    This prevents reloading the model on every function call.
    """
    global reader
    if reader is None:
        print("Initializing OCR reader (this may take a moment on first run)...", flush=True)
        try:
            reader = easyocr.Reader(['ch_sim','en'], gpu=False)
            print("OCR reader loaded successfully", flush=True) 
        except Exception as e:
            print(f"Error loading OCR: {e}", flush=True)
            raise
    return reader

def extract_lines_with_boxes(
    frame_bgr,
    min_confidence=0.05,  # EasyOCR confidence is between 0 and 1
    min_width=20,
    min_height=30,
    min_characters=2,
    source_language="english"
):
    """
    Accepts a BGR frame (NumPy array) directly.
    Returns a list of (text, (x,y,w,h)) for each detected line using OCR.
    """
    # EasyOCR expects RGB
    reader=get_reader()
    processed_frame = preprocess_frame_for_better_precision(frame_bgr)
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
            print(f"🟡 Text: '{clean_text}' | Confidence: {conf:.2f} | Box: ({x}, {y}, {w}, {h})")
            lines.append((clean_text, (x, y, w, h)))
    lines=clean_extracted_lines(lines, CleanConfig(debug=True))
    return lines

