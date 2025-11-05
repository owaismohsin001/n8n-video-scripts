import re

HAN_RE = re.compile(r"[\u4e00-\u9fff]")            # CJK Unified Ideographs
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
VOWEL_RE = re.compile(r"[AEIOUaeiou]")
ALNUM_OR_HAN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
SYMBOL_ONLY_RE = re.compile(r"^\s*[\W_]+\s*$")     # only punctuation/symbols/underscores/space
PUNCT_SIMPLE_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)
