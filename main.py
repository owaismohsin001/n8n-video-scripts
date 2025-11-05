
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


# ============================================================================
# GLOBAL OCR CACHE - Avoid re-processing same frames
# ============================================================================
reader = None
# NEW: Cache stores individual lines with boxes, not concatenated text
ocr_cache: Dict[Tuple[str, int], list] = {}  # (video_path, frame_index) -> [(text, box), ...]


def extract_lines_from_frame_cached(video_path: str, frame_index: int, source_language: str = "english") -> list:
    """
    Extract lines with boxes from a specific frame with caching.
    Returns cached result if available, otherwise performs OCR and caches it.
    
    Returns:
        List of (text, box) tuples, e.g., [("Hello", (x, y, w, h)), ("World", (x, y, w, h))]
    """
    cache_key = (video_path, frame_index)
    
    # Check cache first
    if cache_key in ocr_cache:
        print(f"✓ Cache HIT for frame {frame_index}")
        return ocr_cache[cache_key]
    
    # Cache miss - perform OCR
    print(f"⚙ Cache MISS for frame {frame_index} - performing OCR and cache...")
    frame = get_frame_at_index(video_path, frame_index)
    if frame is None:
        ocr_cache[cache_key] = []
        return []
    
    # Extract individual lines with boxes
    lines = extract_lines_with_boxes(frame)
    print(f"   Extracted {len(lines)} lines from frame {frame_index}")
    ocr_cache[cache_key] = lines
    return lines


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




def is_text_same(video_path: str, frame_index: int, reference_text: str, 
                 similarity_threshold: float, source_language: str = "english") -> Optional[Tuple[bool, str, float]]:
    """
    Check if text at frame_index is similar to reference_text.
    
    Returns:
        Tuple of (is_same, extracted_text, similarity_score) or None if frame unavailable
        
    This avoids duplicate OCR calls by returning the extracted text along with the result.
    """
    # Use cached OCR extraction - now returns individual lines
    lines = extract_lines_from_frame_cached(video_path, frame_index, source_language)
    
    # Convert lines to concatenated text for similarity comparison
    current_text = get_concatenated_text_from_lines(lines)
    
    print(f"   Frame {frame_index}: {len(lines)} lines, text: '{current_text[:50]}...'")
    
    if current_text is None or current_text == "":
        return None
    
    # Calculate similarity
    similarity = text_similarity(reference_text, current_text)
    is_same = similarity >= similarity_threshold
    
    return (is_same, current_text, similarity)


# ============================================================================
# METHOD 2: EXPONENTIAL SEARCH + BINARY SEARCH (OPTIMAL)  for finding the frame where the text changes
# ============================================================================
def find_text_change_optimal(video_path, start_frame_index, source_language="english", similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Find text change using exponential search + binary search.
    This is mathematically optimal with O(log n) complexity.
    
    Phase 1: Exponential search - find range [1, 2, 4, 8, 16, 32, ...]
    Phase 2: Binary search within the range
    
    NOW WITH OPTIMIZATIONS:
    - No duplicate OCR calls (is_text_same returns the text)
    - Uses caching to avoid re-processing frames
    - Cache stores individual lines with boxes
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract reference lines (using cache) and convert to text for comparison
    start_lines = extract_lines_from_frame_cached(video_path, start_frame_index, source_language)
    start_text = get_concatenated_text_from_lines(start_lines)
    
    if start_text is None or start_text == "":
        return -1
    
    print(f"Reference text (frame {start_frame_index}): {len(start_lines)} lines, text: '{start_text[:80]}...'")
    
    frame_checks = 0
    
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
            left = current_index - step + 1
            right = current_index
            print(f"Change detected between frames {left-1} and {right}")
            break
        
        # Double the step size
        step *= 2
        current_index = start_frame_index + step
    else:
        # No change found
        print(f"No text change found. Total frames checked: {frame_checks}")
        return -1
    
    # Phase 2: Binary search within the range
    print(f"\n=== Phase 2: Binary Search [{left}, {right}] ===")
    
    while left < right:
        mid = (left + right) // 2
        result_tuple = is_text_same(video_path, mid, start_text, similarity_threshold, source_language)
        frame_checks += 1
        
        if result_tuple is None:
            break
        
        # Unpack result - NO DUPLICATE OCR!
        is_same, current_text, similarity = result_tuple
        print(f"Frame {mid}: {'SAME' if is_same else 'DIFFERENT'} (similarity={similarity:.3f})")
        
        if is_same:  # Still same text
            left = mid + 1
        else:  # Different text
            right = mid

            
    
    print(f"\n✓ Exact text change frame: {left}")
    print(f"Total frames checked: {frame_checks}")
    print(f"💾 Cache size: {len(ocr_cache)} frames")
    return left


# ============================================================================
# BATCH PROCESSING WITH MULTITHREADING
# ============================================================================



def process_batch_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    batch_id: int,
    source_language: str,
    target_language: str,
    font_path: str,
    font_size: int,
    font_color: str,
    fps: float,
    width: int,
    height: int,
    output_dir: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> str:
    """
    Process a batch of frames: find segments, translate, and overlay.
    This function runs in a separate thread and does EVERYTHING for its batch.
    
    Returns:
        Path to the output video file for this batch
    """
    print(f"\n🔄 [Batch {batch_id}] Processing frames {start_frame} to {end_frame}")
    
    # Step 1: Find text segments within this batch
    segments = []
    current_frame = start_frame
    
    while current_frame < end_frame:
        # Check if current frame has text - get individual lines from cache
        current_lines = extract_lines_from_frame_cached(video_path, current_frame, source_language)
        current_text = get_concatenated_text_from_lines(current_lines)
        
        if not current_text or current_text == "":
            # No text in this frame - find consecutive frames without text
            no_text_start = current_frame
            current_frame += 1
            
            # Find all consecutive frames without text
            while current_frame < end_frame:
                next_lines = extract_lines_from_frame_cached(video_path, current_frame, source_language)
                next_text = get_concatenated_text_from_lines(next_lines)
                if next_text and next_text != "":
                    break  # Found frame with text
                current_frame += 1
            
            # Create segment for all frames without text
            segments.append((no_text_start, current_frame))
            continue
        
        # Find where text changes within this batch
        change_frame = find_text_change_optimal(
            video_path=video_path,
            start_frame_index=current_frame,
            source_language=source_language,
            similarity_threshold=similarity_threshold
        )
        
        if change_frame == -1:
            # No change found, same text till end of batch
            segments.append((current_frame, end_frame))
            break
        elif change_frame >= end_frame:
            # Change is beyond this batch
            segments.append((current_frame, end_frame))
            break
        else:
            # Change found within batch
            segments.append((current_frame, change_frame))
            current_frame = change_frame
    
    print(f"📝 [Batch {batch_id}] Found {len(segments)} segment(s)")
    
    # Step 2: Create output video for this batch
    batch_output_path = os.path.join(output_dir, f"batch_{batch_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(batch_output_path, fourcc, fps, (width, height))
    
    # Step 3: Process each segment (translate and overlay)
    for seg_idx, (seg_start, seg_end) in enumerate(segments, 1):
        print(f"   [Batch {batch_id}] Segment {seg_idx}/{len(segments)}: frames {seg_start}-{seg_end}")
        
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
        translated_lines = []
        for idx, (text, box) in enumerate(lines, 1):
            print(f"      Line {idx}/{len(lines)} - Original: '{text}'")
            translated_text = translate_text(text, target_language=target_language)
            print(f"      Line {idx}/{len(lines)} - Translated: '{translated_text}'")
            translated_lines.append((translated_text, box))
        
        print(f"   [Batch {batch_id}] ✓ Completed translating {len(lines)} lines")
        print(f"   [Batch {batch_id}] Translated lines with boxes: {translated_lines}")
        
        # Overlay translated text for all frames in this segment
        for i in range(seg_start, seg_end):
            frame = get_frame_at_index(video_path, i)
            if frame is None:
                continue
                
            frame_with_overlay = overlay_translated_lines_on_frame(
                frame,
                translated_lines,
                font_path=font_path,
                font_size=font_size,
                font_color=font_color
            )
            out.write(frame_with_overlay)
    
    out.release()
    print(f"✅ [Batch {batch_id}] Completed and saved to {batch_output_path}")
    return batch_output_path


def process_video_parallel(
    video_path: str,
    total_frames: int,
    source_language: str,
    target_language: str,
    font_path: str,
    font_size: int,
    font_color: str,
    fps: float,
    width: int,
    height: int,
    output_dir: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> list:
    """
    Process video in parallel: Each thread finds segments, translates, and overlays.
    
    Returns:
        List of batch video file paths (sorted by batch_id)
    """
    batch_size, num_workers = calculate_optimal_batch_config(total_frames)
    
    # Create batch ranges
    batches = []
    for i in range(0, total_frames, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, total_frames)
        batches.append((batch_start, batch_end, len(batches)))
    
    print(f"\n🚀 Starting parallel processing with {num_workers} workers...")
    print(f"📊 Batches: {len(batches)}, Batch size: {batch_size}, Total frames: {total_frames}")
    
    batch_videos = []
    
    # Process batches in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(
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
            ): (batch_start, batch_end, batch_id)
            for batch_start, batch_end, batch_id in batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_info = future_to_batch[future]
            batch_id = batch_info[2]
            try:
                batch_video_path = future.result()
                batch_videos.append((batch_id, batch_video_path))
                print(f"✓ Batch {batch_id} video collected: {batch_video_path}")
            except Exception as e:
                print(f"❌ Batch {batch_id} failed with error: {e}")
                import traceback
                traceback.print_exc()
    
    # Sort by batch_id to maintain order
    batch_videos.sort(key=lambda x: x[0])
    video_paths = [path for _, path in batch_videos]
    
    print(f"\n✅ Parallel processing complete!")
    print(f"   Total batch videos created: {len(video_paths)}")
    print(f"   Cache size: {len(ocr_cache)} frames")
    
    return video_paths


def merge_video_chunks(batch_video_paths: list, output_path: str, fps: float, width: int, height: int):
    """
    Merge multiple batch video chunks into a single output video.
    
    Args:
        batch_video_paths: List of video file paths (in order)
        output_path: Final output video path
        fps: Frames per second
        width: Video width
        height: Video height
    """
    print(f"\n🔗 Merging {len(batch_video_paths)} video chunks...")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames_written = 0
    
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


def function_overlaying_continuous(video_path, font_path, font_size, out_path=OUTPUT_PATH, 
                                   target_language="English", font_color="black", source_language="english",
                                   use_parallel=True):
    """
    NEW ARCHITECTURE: True parallel processing where each thread does EVERYTHING
    
    Steps:
    1. Extract audio
    2. Each thread processes its batch: find segments → translate → overlay
    3. Merge all batch videos into final output
    4. Add audio back
    
    Args:
        use_parallel: If True, use parallel processing; if False, use legacy sequential
    """
    if not use_parallel:
        print("⚠️ Sequential processing mode is deprecated. Using parallel mode.")
    
    print(f"🎬 Processing video: {video_path}")
    print(f"🚀 Using TRUE PARALLEL processing mode (Thread → Process → Overlay)")
    
    # Extract audio first
    extract_audio(video_path, AUDIO_PATH)
    
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
