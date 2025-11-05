# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import os
import cv2
from utils.overlay_utils import overlay_translated_lines_on_frame  
from utils.translate_utils import translate_lines, translate_text
from utils.ocr.ocr_utils import extract_lines_with_boxes  
from audioUtils import extract_audio,combine_audio_with_video
from utils.vision import get_frame_at_index
from utils.pattern import text_similarity
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional
from constants.index import SIMILARITY_THRESHOLD
from constants.paths import OUTPUT_PATH, AUDIO_PATH
from utils.system_resources import calculate_optimal_batch_config
import json
import time
from datetime import datetime
import threading

# Cython type declarations
cimport cython
from cython.parallel cimport prange, parallel

# ============================================================================
# Type definitions
# ============================================================================
ctypedef dict ocr_cache_type
ctypedef list lines_list_type
ctypedef tuple cache_key_type

# ============================================================================
# GLOBAL OCR CACHE - Avoid re-processing same frames
# ============================================================================
cdef object reader = None
cdef dict ocr_cache = {}  # (video_path, frame_index) -> [(text, box), ...]

# ============================================================================
# LOGGING SYSTEM - Thread-safe batched JSON logging
# ============================================================================
cdef object log_lock = threading.Lock()
cdef str log_dir = "logs"
cdef str current_log_file = None
cdef list log_buffer = []  # In-memory buffer for batched writes
cdef int log_buffer_size = 50  # Flush every N entries
cdef double last_flush_time = 0
cdef double flush_interval = 5.0  # Flush every N seconds if buffer not full

cpdef void init_logging(str session_id = None):
    """Initialize logging system with a new session"""
    global current_log_file, log_dir, log_buffer, last_flush_time
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    current_log_file = os.path.join(log_dir, f"video_processing_{session_id}.json")
    
    # Initialize log file with empty array
    with open(current_log_file, 'w') as f:
        json.dump([], f)
    
    log_buffer = []
    last_flush_time = time.time()
    
    print(f"📝 Logging initialized: {current_log_file}")

cpdef void flush_logs():
    """Flush log buffer to disk"""
    global log_buffer, current_log_file, log_lock, last_flush_time
    
    if not log_buffer or current_log_file is None:
        return
    
    with log_lock:
        try:
            # Read existing logs
            with open(current_log_file, 'r') as f:
                logs = json.load(f)
            
            # Append buffered logs
            logs.extend(log_buffer)
            
            # Write back
            with open(current_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            # Clear buffer
            log_buffer = []
            last_flush_time = time.time()
        except Exception as e:
            print(f"⚠️ Logging error: {e}")

cpdef void log_event(str event_type, dict data, bint immediate = False):
    """
    Thread-safe batched logging to JSON file.
    
    Args:
        event_type: Type of event
        data: Event data dictionary
        immediate: If True, flush immediately (for critical events)
    """
    global current_log_file, log_lock, log_buffer, log_buffer_size, last_flush_time, flush_interval
    
    if current_log_file is None:
        init_logging()
    
    cdef dict log_entry = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": threading.current_thread().ident,
        "thread_name": threading.current_thread().name,
        "event_type": event_type,
        "data": data
    }
    
    # Get current time before lock (time check can happen outside lock)
    cdef double current_time = time.time()
    cdef bint time_expired = (current_time - last_flush_time) >= flush_interval
    
    with log_lock:
        log_buffer.append(log_entry)
        
        # Check if we should flush: immediate flag, buffer full, or time expired
        if immediate or len(log_buffer) >= log_buffer_size or time_expired:
            try:
                # Read existing logs
                with open(current_log_file, 'r') as f:
                    logs = json.load(f)
                
                # Append buffered logs
                logs.extend(log_buffer)
                
                # Write back
                with open(current_log_file, 'w') as f:
                    json.dump(logs, f, indent=2)
                
                # Clear buffer
                log_buffer = []
                last_flush_time = current_time
            except Exception as e:
                print(f"⚠️ Logging error: {e}")

cpdef void finalize_logging():
    """Flush any remaining logs and close logging"""
    flush_logs()

# ============================================================================
# Helper function for numeric comparisons (can use nogil)
# ============================================================================
cdef bint compare_similarity_nogil(double similarity, double threshold) nogil:
    """Pure numeric comparison - can run without GIL"""
    return similarity >= threshold

cdef int calculate_mid_frame(int left, int right) nogil:
    """Pure numeric calculation - can run without GIL"""
    return (left + right) // 2

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

cpdef str clean_extracted_text(str text):
    """
    Cleans OCR-extracted text.
    - Keeps only Chinese characters (and spaces)
    - Removes English letters, digits, and symbols
    - Normalizes whitespace
    
    Note: Uses Python regex, so cannot be nogil
    """
    cdef str clean_text
    # Remove everything except Chinese characters and spaces
    clean_text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    # Normalize multiple spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


cpdef list extract_lines_from_frame_cached(str video_path, int frame_index, str source_language = "english"):
    """
    Extract lines with boxes from a specific frame with caching.
    Returns cached result if available, otherwise performs OCR and caches it.
    
    Returns:
        List of (text, box) tuples, e.g., [("Hello", (x, y, w, h)), ("World", (x, y, w, h))]
    
    Note: Uses Python objects (dict, list, cv2), so cannot be nogil
    """
    cdef tuple cache_key
    cdef object frame
    cdef list lines
    cdef double ocr_start_time
    cdef double ocr_end_time
    
    cache_key = (video_path, frame_index)
    
    # Check cache first
    if cache_key in ocr_cache:
        print(f"✓ Cache HIT for frame {frame_index}")
        return ocr_cache[cache_key]
    
    # Cache miss - perform OCR
    print(f"⚙ Cache MISS for frame {frame_index} - performing OCR and cache...")
    log_event("ocr_cache_miss", {
        "frame_index": frame_index
    })
    
    log_event("frame_read_start", {
        "frame_index": frame_index
    })
    
    ocr_start_time = time.time()
    frame = get_frame_at_index(video_path, frame_index)
    
    if frame is None:
        log_event("frame_read_failed", {
            "frame_index": frame_index,
            "error": "Frame is None"
        })
        ocr_cache[cache_key] = []
        return []
    
    log_event("frame_read_success", {
        "frame_index": frame_index,
        "frame_shape": str(frame.shape) if hasattr(frame, 'shape') else "unknown"
    })
    
    # Extract individual lines with boxes
    log_event("ocr_extraction_start", {
        "frame_index": frame_index
    })
    
    lines = extract_lines_with_boxes(frame)
    ocr_end_time = time.time()
    
    print(f"   Extracted {len(lines)} lines from frame {frame_index}")
    log_event("ocr_extraction_complete", {
        "frame_index": frame_index,
        "lines_count": len(lines),
        "ocr_time_seconds": ocr_end_time - ocr_start_time,
        "lines_preview": [{"text": line[0][:50], "box": line[1]} for line in lines[:3]]
    })
    
    ocr_cache[cache_key] = lines
    return lines


cpdef str get_concatenated_text_from_lines(list lines):
    """
    Helper function to get concatenated text from lines for similarity comparison.
    
    Args:
        lines: List of (text, box) tuples
    
    Returns:
        Concatenated text string (all whitespace removed)
    
    Note: Uses Python list operations, so cannot be nogil
    """
    cdef str text = ""
    cdef str item_text
    cdef tuple item
    cdef int i
    
    if not lines:
        return ""
    
    # Use regular loop instead of generator expression to avoid closure
    for i in range(len(lines)):
        item = lines[i]
        item_text = item[0]
        text += ''.join(item_text.split())
    
    return text


cpdef object is_text_same(str video_path, int frame_index, str reference_text, 
                         double similarity_threshold, str source_language = "english"):
    """
    Check if text at frame_index is similar to reference_text.
    
    Returns:
        Tuple of (is_same, extracted_text, similarity_score) or None if frame unavailable
        
    This avoids duplicate OCR calls by returning the extracted text along with the result.
    
    Note: Uses Python objects and functions, so cannot be nogil
    """
    cdef list lines
    cdef str current_text
    cdef double similarity
    cdef bint is_same
    
    # Use cached OCR extraction - now returns individual lines
    lines = extract_lines_from_frame_cached(video_path, frame_index, source_language)
    
    # Convert lines to concatenated text for similarity comparison
    current_text = get_concatenated_text_from_lines(lines)
    
    print(f"   Frame {frame_index}: {len(lines)} lines, text: '{current_text[:50]}...'")
    
    if current_text is None or current_text == "":
        return None
    
    # Calculate similarity
    similarity = text_similarity(reference_text, current_text)
    
    # Use nogil block for pure numeric comparison
    with nogil:
        is_same = compare_similarity_nogil(similarity, similarity_threshold)
    
    return (is_same, current_text, similarity)


# ============================================================================
# METHOD 2: EXPONENTIAL SEARCH + BINARY SEARCH (OPTIMAL)
# ============================================================================
cpdef int find_text_change_optimal(str video_path, int start_frame_index, 
                                   str source_language = "english", 
                                   double similarity_threshold = SIMILARITY_THRESHOLD):
    """
    Find text change using exponential search + binary search.
    This is mathematically optimal with O(log n) complexity.
    
    Phase 1: Exponential search - find range [1, 2, 4, 8, 16, 32, ...]
    Phase 2: Binary search within the range
    
    NOW WITH OPTIMIZATIONS:
    - No duplicate OCR calls (is_text_same returns the text)
    - Uses caching to avoid re-processing frames
    - Cache stores individual lines with boxes
    - Uses nogil blocks for numeric calculations
    
    Note: Main function uses Python objects, but numeric operations use nogil
    """
    cdef object cap
    cdef int total_frames
    cdef list start_lines
    cdef str start_text
    cdef int frame_checks = 0
    cdef int step
    cdef int current_index
    cdef int left = 0
    cdef int right = 0
    cdef int mid
    cdef object result_tuple
    cdef bint is_same
    cdef str current_text
    cdef double similarity
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract reference lines (using cache) and convert to text for comparison
    start_lines = extract_lines_from_frame_cached(video_path, start_frame_index, source_language)
    start_text = get_concatenated_text_from_lines(start_lines)
    
    if start_text is None or start_text == "":
        return -1
    
    print(f"Reference text (frame {start_frame_index}): {len(start_lines)} lines, text: '{start_text[:80]}...'")
    
    # Phase 1: Exponential search to find upper bound
    print("\n=== Phase 1: Exponential Search ===")
    step = 1
    current_index = start_frame_index + step
    
    while current_index < total_frames:
        result_tuple = is_text_same(video_path, current_index, start_text, similarity_threshold, source_language)
        frame_checks += 1
        
        if result_tuple is None:
            break
        
        # Unpack result - NO DUPLICATE OCR!
        is_same, current_text, similarity = result_tuple
        print(f"Frame {current_index} (step={step}): {'SAME' if is_same else 'DIFFERENT'} (similarity={similarity:.3f})")
        
        if not is_same:  # Found different text
            # The change is between (current_index - step) and current_index
            # Use nogil block for numeric calculation
            with nogil:
                left = current_index - step + 1
                right = current_index
            print(f"Change detected between frames {left-1} and {right}")
            break
        
        # Double the step size - use nogil for numeric operation
        with nogil:
            step *= 2
            current_index = start_frame_index + step
    else:
        # No change found
        print(f"No text change found. Total frames checked: {frame_checks}")
        return -1
    
    # Phase 2: Binary search within the range
    print(f"\n=== Phase 2: Binary Search [{left}, {right}] ===")
    
    while left < right:
        # Use nogil block for numeric calculation
        with nogil:
            mid = calculate_mid_frame(left, right)
        
        result_tuple = is_text_same(video_path, mid, start_text, similarity_threshold, source_language)
        frame_checks += 1
        
        if result_tuple is None:
            break
        
        # Unpack result - NO DUPLICATE OCR!
        is_same, current_text, similarity = result_tuple
        print(f"Frame {mid}: {'SAME' if is_same else 'DIFFERENT'} (similarity={similarity:.3f})")
        
        if is_same:  # Still same text
            with nogil:
                left = mid + 1
        else:  # Different text
            with nogil:
                right = mid
    
    print(f"\n✓ Exact text change frame: {left}")
    print(f"Total frames checked: {frame_checks}")
    print(f"💾 Cache size: {len(ocr_cache)} frames")
    return left


# ============================================================================
# BATCH PROCESSING WITH MULTITHREADING
# ============================================================================

cpdef str process_batch_segment(
    str video_path,
    int start_frame,
    int end_frame,
    int batch_id,
    str source_language,
    str target_language,
    str font_path,
    int font_size,
    str font_color,
    double fps,
    int width,
    int height,
    str output_dir,
    double similarity_threshold = SIMILARITY_THRESHOLD
):
    """
    Process a batch of frames: find segments, translate, and overlay.
    This function runs in a separate thread and does EVERYTHING for its batch.
    
    Returns:
        Path to the output video file for this batch
    
    Note: Uses Python objects (cv2, lists, dicts), but numeric operations use nogil
    """
    cdef double start_time = time.time()
    
    print(f"\n🔄 [Batch {batch_id}] Processing frames {start_frame} to {end_frame}")
    log_event("batch_start", {
        "batch_id": batch_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": end_frame - start_frame
    }, immediate=True)
    
    try:
        return _process_batch_segment_impl(
            video_path, start_frame, end_frame, batch_id,
            source_language, target_language, font_path, font_size,
            font_color, fps, width, height, output_dir,
            similarity_threshold, start_time
        )
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        log_event("batch_fatal_error", {
            "batch_id": batch_id,
            "error": error_msg,
            "traceback": error_trace,
            "start_frame": start_frame,
            "end_frame": end_frame
        }, immediate=True)
        
        print(f"❌ [Batch {batch_id}] FATAL ERROR: {error_msg}")
        print(error_trace)
        raise


cpdef str _process_batch_segment_impl(
    str video_path,
    int start_frame,
    int end_frame,
    int batch_id,
    str source_language,
    str target_language,
    str font_path,
    int font_size,
    str font_color,
    double fps,
    int width,
    int height,
    str output_dir,
    double similarity_threshold,
    double start_time
):
    """
    Internal implementation of batch processing with error handling.
    """
    cdef list segments = []
    cdef int current_frame
    cdef list current_lines
    cdef str current_text
    cdef int no_text_start
    cdef int change_frame
    cdef str batch_output_path
    cdef object out
    cdef object fourcc
    cdef int seg_idx
    cdef int seg_start
    cdef int seg_end
    cdef list lines
    cdef int idx
    cdef tuple item
    cdef str text
    cdef tuple box
    cdef list translated_lines = []
    cdef str translated_text
    cdef int i
    cdef object frame
    cdef object frame_with_overlay
    
    # Step 1: Find text segments within this batch
    log_event("batch_segment_detection_start", {
        "batch_id": batch_id,
        "start_frame": start_frame,
        "end_frame": end_frame
    })
    
    current_frame = start_frame
    
    while current_frame < end_frame:
        # Log every 100th frame to avoid excessive logging
        if current_frame % 100 == 0:
            log_event("batch_frame_check", {
                "batch_id": batch_id,
                "current_frame": current_frame,
                "end_frame": end_frame,
                "progress_percent": int((current_frame - start_frame) * 100 / (end_frame - start_frame))
            })
        
        # Check if current frame has text - get individual lines from cache
        current_lines = extract_lines_from_frame_cached(video_path, current_frame, source_language)
        current_text = get_concatenated_text_from_lines(current_lines)
        
        if not current_text or current_text == "":
            # No text in this frame - find consecutive frames without text
            log_event("batch_no_text_scan_start", {
                "batch_id": batch_id,
                "start_frame": current_frame,
                "end_frame": end_frame
            })
            
            no_text_start = current_frame
            current_frame += 1
            
            # Find all consecutive frames without text
            while current_frame < end_frame:
                # Log every 50th frame to avoid log bloat
                if current_frame % 50 == 0:
                    log_event("batch_no_text_scan_progress", {
                        "batch_id": batch_id,
                        "current_frame": current_frame,
                        "end_frame": end_frame
                    })
                
                current_lines = extract_lines_from_frame_cached(video_path, current_frame, source_language)
                current_text = get_concatenated_text_from_lines(current_lines)
                if current_text and current_text != "":
                    break  # Found frame with text
                current_frame += 1
            
            # Create segment for all frames without text
            segments.append((no_text_start, current_frame))
            log_event("batch_no_text_segment_created", {
                "batch_id": batch_id,
                "segment_start": no_text_start,
                "segment_end": current_frame,
                "frame_count": current_frame - no_text_start
            })
            continue
        
        # Find where text changes within this batch
        log_event("batch_text_change_search_start", {
            "batch_id": batch_id,
            "search_from_frame": current_frame
        })
        
        change_frame = find_text_change_optimal(
            video_path=video_path,
            start_frame_index=current_frame,
            source_language=source_language,
            similarity_threshold=similarity_threshold
        )
        
        log_event("batch_text_change_search_end", {
            "batch_id": batch_id,
            "search_from_frame": current_frame,
            "change_frame": change_frame
        })
        
        if change_frame == -1:
            # No change found, same text till end of batch
            segments.append((current_frame, end_frame))
            log_event("batch_segment_created", {
                "batch_id": batch_id,
                "segment_start": current_frame,
                "segment_end": end_frame,
                "reason": "no_change_found"
            })
            break
        elif change_frame >= end_frame:
            # Change is beyond this batch
            segments.append((current_frame, end_frame))
            log_event("batch_segment_created", {
                "batch_id": batch_id,
                "segment_start": current_frame,
                "segment_end": end_frame,
                "reason": "change_beyond_batch"
            })
            break
        else:
            # Change found within batch
            segments.append((current_frame, change_frame))
            log_event("batch_segment_created", {
                "batch_id": batch_id,
                "segment_start": current_frame,
                "segment_end": change_frame,
                "reason": "change_found"
            })
            current_frame = change_frame
    
    print(f"📝 [Batch {batch_id}] Found {len(segments)} segment(s)")
    log_event("batch_segments_complete", {
        "batch_id": batch_id,
        "total_segments": len(segments),
        "segments": [{"start": s[0], "end": s[1]} for s in segments]
    })
    
    # Step 2: Create output video for this batch
    batch_output_path = os.path.join(output_dir, f"batch_{batch_id}.mp4")
    log_event("batch_video_writer_init", {
        "batch_id": batch_id,
        "output_path": batch_output_path,
        "fps": fps,
        "resolution": f"{width}x{height}"
    })
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(batch_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        log_event("batch_video_writer_error", {
            "batch_id": batch_id,
            "error": "Failed to open VideoWriter"
        })
        raise RuntimeError(f"[Batch {batch_id}] Failed to open VideoWriter")
    
    log_event("batch_video_writer_opened", {
        "batch_id": batch_id,
        "output_path": batch_output_path
    })
    
    # Step 3: Process each segment (translate and overlay)
    for seg_idx, (seg_start, seg_end) in enumerate(segments, 1):
        print(f"   [Batch {batch_id}] Segment {seg_idx}/{len(segments)}: frames {seg_start}-{seg_end}")
        log_event("batch_segment_processing_start", {
            "batch_id": batch_id,
            "segment_idx": seg_idx,
            "total_segments": len(segments),
            "segment_start": seg_start,
            "segment_end": seg_end
        })
        
        # Get lines from cache (already extracted during segment detection)
        lines = extract_lines_from_frame_cached(video_path, seg_start, source_language)
        
        print(f"   [Batch {batch_id}] Retrieved {len(lines)} lines from cache")
        for idx, (text, box) in enumerate(lines, 1):
            print(f"      Cached Line {idx}: '{text}' at position {box}")
        
        # Check if there's any text to translate
        if not lines or len(lines) == 0:
            print(f"   [Batch {batch_id}] No text found in segment - writing frames as-is")
            # No text, write original frames without overlay
            for i in range(seg_start, seg_end):
                frame = get_frame_at_index(video_path, i)
                if frame is None:
                    continue
                out.write(frame)  # Write original frame without overlay
            continue  # Skip to next segment

        # Translate each line INDIVIDUALLY to avoid context mixing
        print(f"   [Batch {batch_id}] Translating {len(lines)} lines individually...")
        log_event("batch_translation_start", {
            "batch_id": batch_id,
            "segment_idx": seg_idx,
            "lines_count": len(lines)
        })
        
        translated_lines = []
        for idx, (text, box) in enumerate(lines, 1):
            print(f"      Line {idx}/{len(lines)} - Original: '{text}'")
            log_event("batch_line_translation_start", {
                "batch_id": batch_id,
                "segment_idx": seg_idx,
                "line_idx": idx,
                "original_text": text
            })
            
            translated_text = translate_text(text, target_language=target_language)
            
            print(f"      Line {idx}/{len(lines)} - Translated: '{translated_text}'")
            log_event("batch_line_translation_end", {
                "batch_id": batch_id,
                "segment_idx": seg_idx,
                "line_idx": idx,
                "translated_text": translated_text
            })
            translated_lines.append((translated_text, box))
        
        print(f"   [Batch {batch_id}] ✓ Completed translating {len(lines)} lines")
        print(f"   [Batch {batch_id}] Translated lines with boxes: {translated_lines}")
        log_event("batch_translation_complete", {
            "batch_id": batch_id,
            "segment_idx": seg_idx,
            "lines_count": len(lines)
        })
        
        # Overlay translated text for all frames in this segment
        log_event("batch_overlay_start", {
            "batch_id": batch_id,
            "segment_idx": seg_idx,
            "frame_range": f"{seg_start}-{seg_end}",
            "frame_count": seg_end - seg_start
        })
        
        for i in range(seg_start, seg_end):
            # Log every 100th frame to avoid excessive logging
            if i % 100 == 0:
                log_event("batch_frame_processing", {
                    "batch_id": batch_id,
                    "segment_idx": seg_idx,
                    "frame_index": i,
                    "frames_remaining": seg_end - i
                })
            
            frame = get_frame_at_index(video_path, i)
            if frame is None:
                log_event("batch_frame_read_error", {
                    "batch_id": batch_id,
                    "frame_index": i,
                    "error": "Frame is None"
                })
                continue
                
            frame_with_overlay = overlay_translated_lines_on_frame(
                frame,
                translated_lines,
                font_path=font_path,
                font_size=font_size,
                font_color=font_color
            )
            out.write(frame_with_overlay)
        
        log_event("batch_overlay_complete", {
            "batch_id": batch_id,
            "segment_idx": seg_idx,
            "frames_written": seg_end - seg_start
        })
    
    out.release()
    
    cdef double elapsed_time = time.time() - start_time
    print(f"✅ [Batch {batch_id}] Completed and saved to {batch_output_path}")
    log_event("batch_complete", {
        "batch_id": batch_id,
        "output_path": batch_output_path,
        "total_segments": len(segments),
        "elapsed_time_seconds": elapsed_time
    }, immediate=True)
    
    return batch_output_path


cpdef list process_video_parallel(
    str video_path,
    int total_frames,
    str source_language,
    str target_language,
    str font_path,
    int font_size,
    str font_color,
    double fps,
    int width,
    int height,
    str output_dir,
    double similarity_threshold = SIMILARITY_THRESHOLD
):
    """
    Process video in parallel: Each thread finds segments, translates, and overlays.
    
    Returns:
        List of batch video file paths (sorted by batch_id)
    
    Note: Uses Python objects (ThreadPoolExecutor, lists), so cannot be nogil
    """
    cdef tuple batch_config
    cdef int batch_size
    cdef int num_workers
    cdef list batches = []
    cdef int i
    cdef int batch_start
    cdef int batch_end
    cdef list batch_videos = []
    cdef object executor
    cdef object future_to_batch
    cdef object future
    cdef tuple batch_info
    cdef int batch_id
    cdef str batch_video_path
    cdef list video_paths
    cdef tuple batch_tuple
    cdef tuple batch_video_tuple
    cdef str path
    cdef int j
    cdef int k
    cdef tuple temp
    cdef int completed_count = 0
    
    batch_config = calculate_optimal_batch_config(total_frames)
    batch_size = batch_config[0]
    num_workers = batch_config[1]
    
    # Create batch ranges
    for i in range(0, total_frames, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, total_frames)
        batches.append((batch_start, batch_end, len(batches)))
    
    print(f"\n🚀 Starting parallel processing with {num_workers} workers...")
    print(f"📊 Batches: {len(batches)}, Batch size: {batch_size}, Total frames: {total_frames}")
    
    log_event("parallel_processing_start", {
        "num_workers": num_workers,
        "total_batches": len(batches),
        "batch_size": batch_size,
        "total_frames": total_frames
    }, immediate=True)
    
    # Process batches in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        log_event("thread_pool_created", {
            "max_workers": num_workers
        })
        
        # Submit all batch jobs - use regular loop instead of dict comprehension to avoid closure
        future_to_batch = {}
        for batch_tuple in batches:
            batch_start, batch_end, batch_id = batch_tuple
            
            log_event("batch_submit_start", {
                "batch_id": batch_id,
                "batch_start": batch_start,
                "batch_end": batch_end
            })
            
            future = executor.submit(
                process_batch_segment,
                video_path,
                batch_start,
                batch_end,
                batch_id,
                source_language,
                target_language,
                font_path,
                font_size,
                font_color,
                fps,
                width,
                height,
                output_dir,
                similarity_threshold
            )
            future_to_batch[future] = (batch_start, batch_end, batch_id)
            
            log_event("batch_submitted", {
                "batch_id": batch_id,
                "total_submitted": len(future_to_batch)
            })
        
        log_event("all_batches_submitted", {
            "total_batches": len(future_to_batch)
        }, immediate=True)
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_batch):
            batch_info = future_to_batch[future]
            batch_id = batch_info[2]
            
            log_event("batch_future_completed", {
                "batch_id": batch_id,
                "completed_count": completed_count + 1,
                "total_batches": len(future_to_batch)
            })
            
            try:
                batch_video_path = future.result()
                batch_videos.append((batch_id, batch_video_path))
                completed_count += 1
                print(f"✓ Batch {batch_id} video collected: {batch_video_path}")
                
                log_event("batch_result_collected", {
                    "batch_id": batch_id,
                    "output_path": batch_video_path,
                    "completed_count": completed_count,
                    "remaining": len(future_to_batch) - completed_count
                }, immediate=True)
            except Exception as e:
                print(f"❌ Batch {batch_id} failed with error: {e}")
                import traceback
                traceback.print_exc()
                
                log_event("batch_error", {
                    "batch_id": batch_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, immediate=True)
    
    # Sort by batch_id to maintain order - manual sort to avoid closure
    for j in range(len(batch_videos)):
        for k in range(j + 1, len(batch_videos)):
            if batch_videos[j][0] > batch_videos[k][0]:
                temp = batch_videos[j]
                batch_videos[j] = batch_videos[k]
                batch_videos[k] = temp
    # Use regular loop instead of list comprehension to avoid closure
    video_paths = []
    for batch_video_tuple in batch_videos:
        path = batch_video_tuple[1]
        video_paths.append(path)
    
    print(f"\n✅ Parallel processing complete!")
    print(f"   Total batch videos created: {len(video_paths)}")
    print(f"   Cache size: {len(ocr_cache)} frames")
    
    # Flush logs after parallel processing completes
    flush_logs()
    
    return video_paths


cpdef merge_video_chunks(list batch_video_paths, str output_path, double fps, int width, int height):
    """
    Merge multiple batch video chunks into a single output video.
    
    Args:
        batch_video_paths: List of video file paths (in order)
        output_path: Final output video path
        fps: Frames per second
        width: Video width
        height: Video height
    
    Note: Uses Python objects (cv2, lists), so cannot be nogil
    """
    cdef object out
    cdef object fourcc
    cdef int total_frames_written = 0
    cdef int idx
    cdef str batch_path
    cdef object cap
    cdef int batch_frames = 0
    cdef bint ret
    cdef object frame
    
    print(f"\n🔗 Merging {len(batch_video_paths)} video chunks...")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for idx, batch_path in enumerate(batch_video_paths, 1):
        print(f"   Merging chunk {idx}/{len(batch_video_paths)}: {batch_path}")
        
        if not os.path.exists(batch_path):
            print(f"   ⚠️ Warning: {batch_path} not found, skipping...")
            continue
        
        # Read batch video and copy frames
        cap = cv2.VideoCapture(batch_path)
        batch_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            batch_frames += 1
            total_frames_written += 1
        
        cap.release()
        print(f"      ✓ Copied {batch_frames} frames")
    
    out.release()
    print(f"✅ Merge complete! Total frames written: {total_frames_written}")
    print(f"   Output: {output_path}")


cpdef function_overlaying_continuous(str video_path, str font_path, int font_size, 
                                     str out_path = OUTPUT_PATH, 
                                     str target_language = "English", 
                                     str font_color = "black", 
                                     str source_language = "english",
                                     bint use_parallel = True):
    """
    NEW ARCHITECTURE: True parallel processing where each thread does EVERYTHING
    
    Steps:
    1. Extract audio
    2. Each thread processes its batch: find segments → translate → overlay
    3. Merge all batch videos into final output
    4. Add audio back
    
    Args:
        use_parallel: If True, use parallel processing; if False, use legacy sequential
    
    Note: Uses Python objects (cv2, os, lists), so cannot be nogil
    """
    cdef object cap
    cdef double fps
    cdef int width
    cdef int height
    cdef int total_frames
    cdef str temp_dir
    cdef list batch_video_paths
    cdef str batch_video
    
    # Initialize logging
    init_logging()
    
    if not use_parallel:
        print("⚠️ Sequential processing mode is deprecated. Using parallel mode.")
    
    print(f"🎬 Processing video: {video_path}")
    print(f"🚀 Using TRUE PARALLEL processing mode (Thread → Process → Overlay)")
    
    log_event("video_processing_start", {
        "video_path": video_path,
        "font_path": font_path,
        "font_size": font_size,
        "target_language": target_language,
        "source_language": source_language,
        "font_color": font_color,
        "output_path": out_path
    }, immediate=True)
    
    # Extract audio first
    log_event("audio_extraction_start", {
        "video_path": video_path,
        "audio_path": AUDIO_PATH
    })
    
    extract_audio(video_path, AUDIO_PATH)
    
    log_event("audio_extraction_complete", {
        "audio_path": AUDIO_PATH
    })
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"📹 Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Create temporary directory for batch videos
    temp_dir = os.path.join(os.path.dirname(out_path), "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"📁 Temporary batch directory: {temp_dir}")
    
    # STEP 1: Process video in parallel (each thread does EVERYTHING)
    print("\n" + "="*60)
    print("STEP 1: Parallel Processing (Find + Translate + Overlay)")
    print("="*60)
    
    batch_video_paths = process_video_parallel(
        video_path=video_path,
        total_frames=total_frames,
        source_language=source_language,
        target_language=target_language,
        font_path=font_path,
        font_size=font_size,
        font_color=font_color,
        fps=fps,
        width=width,
        height=height,
        output_dir=temp_dir,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    if not batch_video_paths:
        print("⚠️ No batch videos created!")
        return
    
    # STEP 2: Merge all batch videos
    print("\n" + "="*60)
    print("STEP 2: Merging batch videos")
    print("="*60)
    
    merge_video_chunks(
        batch_video_paths=batch_video_paths,
        output_path=out_path,
        fps=fps,
        width=width,
        height=height
    )
    
    # STEP 3: Combine with audio
    print("\n" + "="*60)
    print("STEP 3: Adding audio")
    print("="*60)
    combine_audio_with_video(
        silent_video_path=out_path, 
        audio_path=AUDIO_PATH, 
        combined_audio_video_path=out_path
    )
    
    # STEP 4: Cleanup temporary files
    print("\n" + "="*60)
    print("STEP 4: Cleanup")
    print("="*60)
    
    try:
        # Remove audio file
        if os.path.exists(AUDIO_PATH):
            os.remove(AUDIO_PATH)
            print(f"🗑️ Removed temporary audio file")
        
        # Remove batch videos
        for batch_video in batch_video_paths:
            if os.path.exists(batch_video):
                os.remove(batch_video)
                print(f"🗑️ Removed {os.path.basename(batch_video)}")
        
        # Remove temp directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            print(f"🗑️ Removed temporary directory")
            
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   Total frames: {total_frames}")
    print(f"   Batch videos created: {len(batch_video_paths)}")
    print(f"   OCR cache size: {len(ocr_cache)} frames")
    print(f"   Output: {out_path}")
    
    log_event("video_processing_complete", {
        "total_frames": total_frames,
        "batch_videos_created": len(batch_video_paths),
        "ocr_cache_size": len(ocr_cache),
        "output_path": out_path,
        "log_file": current_log_file
    }, immediate=True)
    
    # Finalize logging - flush any remaining logs
    finalize_logging()
    
    print(f"\n📝 Detailed logs saved to: {current_log_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay translations on video frames with PARALLEL processing support."
    )
    parser.add_argument("--video", dest="video_path", required=True, help="Path to input video")
    parser.add_argument("--font", dest="font_path", default=None, help="Path to TTF font (optional)")
    parser.add_argument("--fontSize", dest="font_size", default="35", help="Font size (int)")
    parser.add_argument("--out", dest="out_path", default=OUTPUT_PATH, help="Output video path")
    parser.add_argument("--targetLang", dest="target_language", default="ch_sim", help="Target language for translation")
    parser.add_argument("--fontColor", dest="font_color", default="black", help="Font color for translation overlay")
    parser.add_argument("--sourceLang", dest="source_language", default="english", help="Source language of the video")
    parser.add_argument(
        "--parallel", 
        dest="use_parallel", 
        action="store_true", 
        default=True,
        help="Use parallel batch processing (default: True)"
    )
    parser.add_argument(
        "--sequential", 
        dest="use_parallel", 
        action="store_false",
        help="Use sequential processing (legacy mode)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VIDEO TRANSLATION WITH OCR")
    print("="*60)
    print(f"Mode: {'PARALLEL (Multi-threaded)' if args.use_parallel else 'SEQUENTIAL (Legacy)'}")
    print("="*60 + "\n")
    
    function_overlaying_continuous(
        video_path=args.video_path,
        font_path=args.font_path,
        font_size=int(args.font_size),
        out_path=args.out_path,
        target_language=args.target_language,
        font_color=args.font_color,
        source_language=args.source_language,
        use_parallel=args.use_parallel
    )


#"en", "de", "es"
