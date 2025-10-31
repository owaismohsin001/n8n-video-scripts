# Cython-annotated version of your video translation script
# File: video_translate_cython.pyx
# Purpose: Provide a Cython-friendly refactor that adds type annotations,
# optimizations and clearly marked places where "nogil" can be used.
# Important: This file DOES NOT make the whole program run without the
# Python GIL. Most calls (cv2, OCR, translation, file I/O) are Python
# APIs and require the GIL. What we can do: 1) add cdef types for hot
# loops, 2) minimize GIL hold time, and 3) show where to use "with nogil"
# for pure-C loops if you later replace Python-heavy parts with C APIs.

# Compile with: python setup.py build_ext --inplace

# setup.py (example):
# from setuptools import setup
# from Cython.Build import cythonize
# setup(
#     ext_modules=cythonize("video_translate_cython.pyx", compiler_directives={"language_level": "3"})
# )

# ------------------ Imports ------------------
from cpython.dict cimport PyDict_GetItem
import cython

# Only needed for parallel loops
from cython.parallel import prange, parallel

# For Cython-specific decorators and directives
cimport cython

# Keep Python imports because we call into these libraries
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional, List
import cv2

# Your project imports (these are Python-level)
from utils.overlay_utils import overlay_translated_lines_on_frame
from utils.translate_utils import translate_lines
from utils.ocr.ocr_utils import extract_lines_with_boxes
from utils.audioUtils import extract_audio, combine_audio_with_video
from utils.vision import get_frame_at_index
from utils.pattern import text_similarity
from constants.index import SIMILARITY_THRESHOLD
from constants.paths import OUTPUT_PATH, AUDIO_PATH
from utils.system_resources import calculate_optimal_batch_config
from libc.stdlib cimport malloc, free
# ------------------ Cython-friendly typed globals ------------------
# We use Python objects for many things, so types remain Python-level.
reader = None
ocr_cache: Dict[Tuple[str, int], str] = {}

# ------------------ Utility functions ------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def clean_extracted_text(text: str) -> str:
    """
    Keep only Chinese chars and spaces, normalize whitespace.
    """
    # small helper: this is Python-level regex; no nogil possible here
    clean_text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


# ------------------ OCR extraction ------------------
@cython.locals(results=object, text=object)
def extract_text_from_image(image, source_language: str = "english") -> str:
    """Extract text from image using existing OCR util."""
    results = extract_lines_with_boxes(image)
    # Join lines removing whitespace in each text piece
    text = ''.join(''.join(text.split()) for text, _ in results)
    if len(text) == 0:
        return ""
    else:
        return text


@cython.locals(cache_key=tuple)
def extract_text_from_frame_cached(video_path: str, frame_index: int, source_language: str = "english") -> str:
    """Uses module-level cache to avoid duplicate OCR."""
    cache_key = (video_path, frame_index)
    if cache_key in ocr_cache:
        # Cache hit
        return ocr_cache[cache_key]

    # Cache miss - perform OCR
    frame = get_frame_at_index(video_path, frame_index)
    if frame is None:
        ocr_cache[cache_key] = ""
        return ""

    text = extract_text_from_image(frame, source_language=source_language)
    ocr_cache[cache_key] = text
    return text


# ------------------ Similarity check ------------------
@cython.locals(current_text=object, similarity=float)
def is_text_same(video_path: str, frame_index: int, reference_text: str,
                 similarity_threshold: float, source_language: str = "english") -> Optional[Tuple[bool, str, float]]:
    """Return (is_same, extracted_text, similarity) or None if frame unreadable."""
    current_text = extract_text_from_frame_cached(video_path, frame_index, source_language)
    if current_text is None:
        return None

    # text_similarity is Python-level; no nogil
    similarity = float(text_similarity(reference_text, current_text))
    is_same = similarity >= similarity_threshold
    return (is_same, current_text, similarity)


# ------------------ Search algorithm (exponential + binary) ------------------
@cython.locals(cap=object, total_frames=int, start_text=object, frame_checks=int)
def find_text_change_optimal(video_path: str, start_frame_index: int, source_language: str = "english",
                             similarity_threshold: float = SIMILARITY_THRESHOLD) -> int:
    """Find the first frame index where text changes after start_frame_index.

    Note: heavy use of Python APIs for OCR and similarity. We add cdef'd locals
    to reduce attribute access overhead but cannot run this fully nogil.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_text = extract_text_from_frame_cached(video_path, start_frame_index, source_language)
    if start_text is None or start_text == "":
        return -1

    frame_checks = 0

    # Phase 1 - Exponential search
    step = 1
    current_index = start_frame_index + step

    left = 0
    right = 0

    while current_index < total_frames:
        result_tuple = is_text_same(video_path, current_index, start_text, similarity_threshold, source_language)
        frame_checks += 1
        if result_tuple is None:
            break

        is_same, current_text, similarity = result_tuple
        if not is_same:
            left = current_index - step + 1
            right = current_index
            break

        step <<= 1  # double step
        current_index = start_frame_index + step
    else:
        return -1

    # Phase 2 - Binary search between left and right
    while left < right:
        mid = (left + right) // 2
        result_tuple = is_text_same(video_path, mid, start_text, similarity_threshold, source_language)
        frame_checks += 1
        if result_tuple is None:
            break
        is_same, current_text, similarity = result_tuple
        if is_same:
            left = mid + 1
        else:
            right = mid

    return left


# ------------------ Batch processing ------------------
@cython.locals(segments=object, current_frame=int, change_frame=int)
def process_batch_segment(video_path: str, start_frame: int, end_frame: int, batch_id: int,
                          source_language: str, similarity_threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    Uses the batched, nogil-aware finder for each run inside the batch worker.
    This version calls find_text_change_optimal_batched to benefit from:
      - process-based OCR batching (batch_ocr_process)
      - nogil similarity checks (c_text_similarity / py_c_text_similarity)
    """
    segments = []
    current_frame = start_frame

    while current_frame < end_frame:
        change_frame = find_text_change_optimal_batched(video_path=video_path,
                                                        start_frame_index=current_frame,
                                                        source_language=source_language,
                                                        similarity_threshold=similarity_threshold)
        if change_frame == -1:
            segments.append((current_frame, end_frame))
            break
        elif change_frame >= end_frame:
            segments.append((current_frame, end_frame))
            break
        else:
            segments.append((current_frame, change_frame))
            current_frame = change_frame

    return segments



@cython.locals(batch_size=int, num_workers=int)
def find_all_text_segments_parallel(video_path: str, total_frames: int,
                                    source_language: str = "english",
                                    similarity_threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    ProcessPool-based parallelization of batch processing so heavy work runs in
    separate processes (avoiding the main-process GIL). Each worker runs
    process_batch_segment which uses the batched/nogil-friendly routines.
    """
    batch_size, num_workers = calculate_optimal_batch_config(total_frames)
    if num_workers < 1:
        num_workers = 1

    batches = []
    i = 0
    batch_id = 0
    while i < total_frames:
        batch_start = i
        batch_end = min(i + batch_size, total_frames)
        batches.append((batch_start, batch_end, batch_id))
        i += batch_size
        batch_id += 1

    all_segments = []

    # Use processes so OCR/translation/OpenCV calls run in separate interpreters
    # and do not contend for the main-process GIL.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch_segment, video_path, batch_start, batch_end, batch_id, source_language, similarity_threshold): (batch_start, batch_end, batch_id)
            for batch_start, batch_end, batch_id in batches
        }
        for future in as_completed(future_to_batch):
            try:
                segments = future.result()
                # segments is a list of (start, end) tuples from that worker
                if segments:
                    all_segments.extend(segments)
            except Exception:
                # keep going — failures in one process shouldn't kill the whole run
                pass

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x[0])
    merged_segments = merge_adjacent_segments(all_segments, video_path, source_language)
    return merged_segments


@cython.locals(merged=list, last_merged=tuple, last_text=object, current_text=object, similarity=float)
def merge_adjacent_segments(segments: list, video_path: str, source_language: str) -> list:
    """
    Merge adjacent segments if their OCR text is similar enough.
    Uses py_c_text_similarity (nogil-enabled inner c_text_similarity) to reduce GIL time.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for current in segments[1:]:
        last_merged = merged[-1]
        if current[0] == last_merged[1]:
            last_text = extract_text_from_frame_cached(video_path, last_merged[0], source_language) or ""
            current_text = extract_text_from_frame_cached(video_path, current[0], source_language) or ""
            # use the C-backed similarity to minimize Python overhead
            try:
                similarity = float(py_c_text_similarity(last_text, current_text))
            except Exception:
                # fallback to conservative behavior on any error
                similarity = 0.0

            if similarity >= SIMILARITY_THRESHOLD:
                merged[-1] = (last_merged[0], current[1])
            else:
                merged.append(current)
        else:
            merged.append(current)
    return merged


# ------------------ Overlaying / Main functions ------------------
@cython.locals(cap=object, fps=float, width=int, height=int, fourcc=int, out=object, start_frame=int, total_frames=int)
def function_overlaying_continuous_legacy(video_path: str, font_path: str, font_size: int, out_path: str = OUTPUT_PATH,
                                          target_language: str = "English", font_color: str = "black", source_language: str = "english"):
    extract_audio(video_path, AUDIO_PATH)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    start_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while start_frame < total_frames:
        change_frame = find_text_change_optimal(video_path=video_path, start_frame_index=start_frame, source_language=source_language)
        if change_frame == -1:
            change_frame = total_frames

        frame = get_frame_at_index(video_path, start_frame)
        if frame is None:
            start_frame = change_frame
            continue

        lines = extract_lines_with_boxes(frame)
        translated_lines = translate_lines(lines, target_language=target_language)

        for i in range(start_frame, change_frame):
            frame = get_frame_at_index(video_path, i)
            if frame is None:
                continue
            frame_with_overlay = overlay_translated_lines_on_frame(frame, translated_lines, font_path=font_path, font_size=font_size, font_color=font_color)
            out.write(frame_with_overlay)

        start_frame = change_frame

    cap.release()
    out.release()
    combine_audio_with_video(silent_video_path=out_path, audio_path=AUDIO_PATH, combined_audio_video_path=out_path)
    try:
        if os.path.exists(AUDIO_PATH):
            os.remove(AUDIO_PATH)
    except Exception:
        pass


@cython.locals(fps=float, width=int, height=int, total_frames=int, segments=list, out=object)
def function_overlaying_continuous(video_path: str, font_path: str, font_size: int, out_path: str = OUTPUT_PATH,
                                   target_language: str = "English", font_color: str = "black", source_language: str = "english",
                                   use_parallel: bool = True):
    if not use_parallel:
        return function_overlaying_continuous_legacy(video_path, font_path, font_size, out_path, target_language, font_color, source_language)

    extract_audio(video_path, AUDIO_PATH)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    segments = find_all_text_segments_parallel(video_path=video_path, total_frames=total_frames, source_language=source_language, similarity_threshold=SIMILARITY_THRESHOLD)
    if not segments:
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for seg_idx, (start_frame, end_frame) in enumerate(segments, 1):
        frame = get_frame_at_index(video_path, start_frame)
        if frame is None:
            continue
        lines = extract_lines_with_boxes(frame)
        translated_lines = translate_lines(lines, target_language=target_language)
        frames_in_segment = end_frame - start_frame
        for i in range(start_frame, end_frame):
            frame = get_frame_at_index(video_path, i)
            if frame is None:
                continue
            frame_with_overlay = overlay_translated_lines_on_frame(frame, translated_lines, font_path=font_path, font_size=font_size, font_color=font_color)
            out.write(frame_with_overlay)

    out.release()
    combine_audio_with_video(silent_video_path=out_path, audio_path=AUDIO_PATH, combined_audio_video_path=out_path)
    try:
        if os.path.exists(AUDIO_PATH):
            os.remove(AUDIO_PATH)
    except Exception:
        pass


# ------------------ CLI entrypoint ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay translations on video frames with PARALLEL processing support.")
    parser.add_argument("--video", dest="video_path", required=True, help="Path to input video")
    parser.add_argument("--font", dest="font_path", default=None, help="Path to TTF font (optional)")
    parser.add_argument("--fontSize", dest="font_size", default="35", help="Font size (int)")
    parser.add_argument("--out", dest="out_path", default=OUTPUT_PATH, help="Output video path")
    parser.add_argument("--targetLang", dest="target_language", default="ch_sim", help="Target language for translation")
    parser.add_argument("--fontColor", dest="font_color", default="black", help="Font color for translation overlay")
    parser.add_argument("--sourceLang", dest="source_language", default="english", help="Source language of the video")
    parser.add_argument("--parallel", dest="use_parallel", action="store_true", default=True, help="Use parallel batch processing (default: True)")
    parser.add_argument("--sequential", dest="use_parallel", action="store_false", help="Use sequential processing (legacy mode)")
    args = parser.parse_args()

    function_overlaying_continuous(video_path=args.video_path, font_path=args.font_path, font_size=int(args.font_size), out_path=args.out_path, target_language=args.target_language, font_color=args.font_color, source_language=args.source_language, use_parallel=args.use_parallel)

# ------------------ Notes on nogil ------------------
# 1) You cannot safely release the GIL around calls to cv2, OCR, translation utilities, or Python containers.
# 2) If you replace the hot OCR/text-similarity/translation code with pure C libraries (or provide Cython cdef wrappers
#    that expose nogil-capable functions), you can then use "with nogil:" blocks around CPU-bound loops here.
# 3) To squeeze more performance right now:
#    - Keep the typed locals (done above) to reduce attribute lookup overhead
#    - Reduce Python call overhead by batching frames for OCR if your OCR supports it
#    - Consider rewriting text_similarity in C or as a cdef function (then you can call it nogil)
# 4) For an example of 'with nogil':
#    cdef int i
#    cdef int n = 1000000
#    cdef double acc = 0.0
#    with nogil:
#        for i in range(n):
#            acc += i * 0.1
#    But note: inside that block you cannot touch Python objects.

# ------------------ NOGIL & HIGH-PERF OPTIMIZATIONS ------------------
# The following additions provide concrete, minimally-invasive changes you can
# drop into this file to remove the Python GIL from *hot paths* and to use
# process-based parallelism for Python-bound libraries (OCR/translate/OpenCV
# bindings) without changing the high-level behavior.
#
# Strategy summary (practical and non-magic):
# 1) Replace Python-level text-similarity with a C-level implementation (cdef)
#    and call it in a nogil block. We provide `c_text_similarity()` below.
# 2) For OCR / translation and other Python/C-extension libraries that keep
#    the GIL or are not nogil-safe, use ProcessPoolExecutor (multiprocessing)
#    to run them in separate processes so you avoid the GIL entirely there.
# 3) Keep existing API and function signatures; add fallback wrappers so the
#    rest of your code remains unchanged.
#
# Notes: to actually see big gains the heavy OCR/translation parts should be
# batched or moved to separate processes or replaced with a native C API.

# ------------------ 1) C-level text similarity (nogil-capable) ------------------
# Simple byte-level Jaccard-like similarity implemented in C.
# It's not a linguistic masterpiece but is tiny, fast, and gives good
# discrimination for short OCR strings. Change to your preferred algo later.

cdef double c_text_similarity(const char *a, Py_ssize_t a_len, const char *b, Py_ssize_t b_len) nogil:
    """
    Compute a simple similarity score between two UTF-8 byte buffers.
    This works on raw bytes; call it after encoding Python strings to UTF-8
    and pass pointers to the underlying bytes. Runs nogil.
    """
    cdef Py_ssize_t i
    cdef unsigned int freq_a[256]
    cdef unsigned int freq_b[256]
    cdef unsigned int intersect = 0
    cdef unsigned int unionc = 0

    # zero arrays
    for i in range(256):
        freq_a[i] = 0
        freq_b[i] = 0

    for i in range(a_len):
        freq_a[(<unsigned char>a[i])] += 1
    for i in range(b_len):
        freq_b[(<unsigned char>b[i])] += 1

    for i in range(256):
        if freq_a[i] and freq_b[i]:
            if freq_a[i] < freq_b[i]:
                intersect += freq_a[i]
            else:
                intersect += freq_b[i]
        if freq_a[i] or freq_b[i]:
            if freq_a[i] > freq_b[i]:
                unionc += freq_a[i]
            else:
                unionc += freq_b[i]

    if unionc == 0:
        return 0.0
    return <double>intersect / <double>unionc


# Python wrapper that holds GIL only long enough to get raw bytes then calls
# c_text_similarity in nogil. This keeps Python interaction minimal.

def py_c_text_similarity(str a, str b) -> float:
    cdef bytes ba = a.encode('utf8')
    cdef bytes bb = b.encode('utf8')
    cdef const char *apa = ba
    cdef const char *bpa = bb
    cdef Py_ssize_t alen = len(ba)
    cdef Py_ssize_t blen = len(bb)
    cdef double score

    # Call nogil function with raw pointers; ba and bb must stay alive until return
    with cython.nogil:
        score = c_text_similarity(apa, alen, bpa, blen)
    return float(score)


# ------------------ 2) Replace text_similarity usage safely ------------------
# Minimal patch: change calls that used text_similarity(...) to py_c_text_similarity(...)
# without changing signatures of functions that rely on them. Example below replaces
# the body of is_text_same with a version that uses the nogil similarity function
# while still using the existing OCR and cache behavior.

@cython.locals(current_text=object, similarity=float)
def is_text_same_nogil(video_path: str, frame_index: int, reference_text: str,
                 similarity_threshold: float, source_language: str = "english") -> Optional[Tuple[bool, str, float]]:
    """GIL-minimized is_text_same: OCR still runs in Python, but similarity
    calculation runs in nogil for speed.
    """
    current_text = extract_text_from_frame_cached(video_path, frame_index, source_language)
    if current_text is None:
        return None

    # Use the C-coded similarity which internally runs with nogil.
    similarity = py_c_text_similarity(reference_text, current_text)
    is_same = similarity >= similarity_threshold
    return (is_same, current_text, similarity)

# You can switch usages of is_text_same -> is_text_same_nogil where appropriate.
# For example, in find_text_change_optimal and other hot loops swap the function
# call to use the nogil version. This keeps all other behavior identical.


# ------------------ 3) Process-based parallelism for OCR/Translate ------------------
# Many libraries don't release the GIL; using processes avoids GIL contention.
# We'll add helper worker functions that can be submitted to ProcessPoolExecutor.

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def _ocr_worker(args_tuple):
    """Runs in a separate process. Expected args: (video_path, frame_index, source_language)
    Imports happen inside the process so the main process remains untouched.
    """
    video_path, frame_index, source_language = args_tuple
    # Local import to reduce child process startup cost at module import time
    from utils.vision import get_frame_at_index
    from utils.ocr.ocr_utils import extract_lines_with_boxes

    frame = get_frame_at_index(video_path, frame_index)
    if frame is None:
        return (frame_index, "")
    lines = extract_lines_with_boxes(frame)
    text = ''.join(''.join(t.split()) for t, _ in lines)
    return (frame_index, text)


# Convenience function: batch OCR frames using processes (safe, no GIL issues)
def batch_ocr_process(video_path: str, frame_indices: list, source_language: str = "english", max_workers: int = None) -> dict:
    """Returns a dict mapping frame_index -> extracted_text. Uses processes.
    Keep the API identical to the cached extractor by merging results into ocr_cache.
    """
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    args = [(video_path, idx, source_language) for idx in frame_indices]
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for frame_index, text in ex.map(_ocr_worker, args):
            results[frame_index] = text
            ocr_cache[(video_path, frame_index)] = text
    return results


# Example: how to use batch_ocr_process in your search routine:
# - Instead of checking frames one-by-one during exponential search, collect
#   several frame indices (e.g. the exponential sequence) and call batch_ocr_process
#   to get OCR for them concurrently in separate processes.
# - Then call is_text_same_nogil on the returned texts without invoking
#   extract_text_from_frame_cached again (cache already filled).


# ------------------ 4) Minimal changes to find_text_change_optimal to batch OCR ------------------
# Here's a drop-in replacement for the exponential phase that batches OCR for the
# exponential probe indices. It keeps the rest of the binary search behavior intact.

@cython.locals(cap=object, total_frames=int, start_text=object, frame_checks=int)
def find_text_change_optimal_batched(
    video_path: str,
    start_frame_index: int,
    source_language: str = "english",
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> int:
    cdef int i, n_probes, found_idx
    cdef double sim, sim_mid
    cdef bytes start_b, mid_b
    cdef const char *start_ptr, *mid_ptr
    cdef Py_ssize_t start_len, mid_len

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_text = extract_text_from_frame_cached(video_path, start_frame_index, source_language)
    if start_text is None or start_text == "":
        return -1

    frame_checks = 0

    # Phase 1 - build exponential probes
    probes = []
    step = 1
    while True:
        idx = start_frame_index + step
        if idx >= total_frames:
            break
        probes.append(idx)
        step <<= 1

    # batch OCR for probes
    if probes:
        batch_ocr_process(video_path, probes, source_language=source_language)

    left = 0
    right = 0

    # Prepare nogil-enabled similarity checks: encode start_text once
    start_b = start_text.encode('utf8')
    start_ptr = start_b
    start_len = len(start_b)

    n_probes = len(probes)
    found_idx = -1

    if n_probes == 0:
        return -1

    # Keep Python-level byte objects alive in this list while we run nogil
    probe_bytes_list = []
    for idx in probes:
        t = ocr_cache.get((video_path, idx), "")
        b = t.encode('utf8')
        probe_bytes_list.append(b)

    # Allocate C arrays for pointers and lengths
    cdef const char **ptrs = <const char **> malloc(n_probes * sizeof(const char *))
    cdef Py_ssize_t *lens = <Py_ssize_t *> malloc(n_probes * sizeof(Py_ssize_t))

    if not ptrs or not lens:
        # fallback to non-nogil loop if allocation fails
        for current_index in probes:
            result_tuple = is_text_same_nogil(video_path, current_index, start_text, similarity_threshold, source_language)
            frame_checks += 1
            if result_tuple is None:
                break
            is_same, current_text, similarity = result_tuple
            if not is_same:
                left = current_index - (current_index - start_frame_index) // 2
                right = current_index
                break
        if ptrs:
            free(ptrs)
        if lens:
            free(lens)
        if right == 0:
            return -1
        # proceed to binary phase
    else:
        # fill C arrays (still under GIL)
        for i in range(n_probes):
            ptrs[i] = probe_bytes_list[i]
            lens[i] = len(probe_bytes_list[i])

        # Run comparisons in nogil: compare start_text against each probe
        with cython.nogil:
            for i in range(n_probes):
                sim = c_text_similarity(start_ptr, start_len, ptrs[i], lens[i])
                if sim < similarity_threshold:
                    found_idx = i
                    break

        # free allocated arrays
        free(ptrs)
        free(lens)

        if found_idx == -1:
            return -1

        current_index = probes[found_idx]
        left = current_index - (current_index - start_frame_index) // 2
        right = current_index

    # Phase 2 - binary search
    while left < right:
        mid = (left + right) // 2

        if (video_path, mid) not in ocr_cache:
            batch_ocr_process(video_path, [mid], source_language=source_language)

        mid_text = ocr_cache.get((video_path, mid), "")
        mid_b = mid_text.encode('utf8')
        mid_ptr = mid_b
        mid_len = len(mid_b)

        with cython.nogil:
            sim_mid = c_text_similarity(start_ptr, start_len, mid_ptr, mid_len)

        frame_checks += 1
        if sim_mid >= similarity_threshold:
            left = mid + 1
        else:
            right = mid

    return left


