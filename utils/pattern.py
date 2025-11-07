from difflib import SequenceMatcher # returns np array frame
import re


def text_similarity(a, b):
    a = a or ""
    b = b or ""
    print(f"Comparing texts:\nA: {a}\nB: {b}")
    total = len(a) + len(b)
    if total == 0:
        return 1.0  # nothing to compare
    matcher = SequenceMatcher(None, a, b)
    print("Matcher :", matcher)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    print(match)
    common = match.size
    print(common,"common")
    print(2 * common / total)
    return 2 * common / total


def clean_extracted_text(text):
    """
    Cleans OCR-extracted text.
    - Keeps only Chinese characters (and spaces)
    - Removes English letters, digits, and symbols
    - Normalizes whitespace
    """
    # Remove everything except Chinese characters and spaces
    clean_text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    # Normalize multiple spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_concatenated_text_from_lines(lines: list) -> str:
    """
    Helper function to get concatenated text from lines for similarity comparison.
    
    Args:
        lines: List of (text, box) tuples
    
    Returns:
        Concatenated text string (all whitespace removed)
    """
    if not lines:
        return ""
    text = ''.join(''.join(text.split()) for text, _ in lines)
    return text