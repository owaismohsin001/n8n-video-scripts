
import os
import cv2
from utils.overlay_utils import overlay_translated_lines_on_frame  
from utils.translate_utils import translate_lines
from utils.ocr_utils import extract_lines_with_boxes  
from utils.audioUtils import extract_audio,combine_audio_with_video
from utils.vision import get_frame_at_index
from utils.pattern import text_similarity
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from typing import Dict, Tuple, Optional

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
ocr_cache: Dict[Tuple[str, int], str] = {}  # (video_path, frame_index) -> extracted_text



def extract_text_from_image(image, source_language="english"):
    """Extract text from image using OCR"""
    results = extract_lines_with_boxes(image)
    print(results, "results")
    text = ''.join(''.join(text.split()) for text, _ in results)
    print(len(text), "length of text")
    if len(text) == 0:
        print("hidden text:", text)
        return ""   
    else:
        print("text:", text)
        return text


def extract_text_from_frame_cached(video_path: str, frame_index: int, source_language: str = "english") -> str:
    """
    Extract text from a specific frame with caching.
    Returns cached result if available, otherwise performs OCR and caches it.
    """
    cache_key = (video_path, frame_index)
    
    # Check cache first
    if cache_key in ocr_cache:
        print(f"✓ Cache HIT for frame {frame_index}")
        return ocr_cache[cache_key]
    
    # Cache miss - perform OCR
    print(f"⚙ Cache MISS for frame {frame_index} - performing OCR...")
    frame = get_frame_at_index(video_path, frame_index)
    if frame is None:
        ocr_cache[cache_key] = ""
        return ""
    
    text = extract_text_from_image(frame, source_language=source_language)
    ocr_cache[cache_key] = text
    return text




def is_text_same(video_path: str, frame_index: int, reference_text: str, 
                 similarity_threshold: float, source_language: str = "english") -> Optional[Tuple[bool, str, float]]:
    """
    Check if text at frame_index is similar to reference_text.
    
    Returns:
        Tuple of (is_same, extracted_text, similarity_score) or None if frame unavailable
        
    This avoids duplicate OCR calls by returning the extracted text along with the result.
    """
    # Use cached OCR extraction
    current_text = extract_text_from_frame_cached(video_path, frame_index, source_language)
    
    if current_text is None:
        return None
    
    # Calculate similarity
    similarity = text_similarity(reference_text, current_text)
    is_same = similarity >= similarity_threshold
    
    return (is_same, current_text, similarity)


# ============================================================================
# METHOD 2: EXPONENTIAL SEARCH + BINARY SEARCH (OPTIMAL)  for finding the frame where the text changes
# ============================================================================
def find_text_change_optimal(video_path, start_frame_index, source_language="english", similarity_threshold=0.85):
    """
    Find text change using exponential search + binary search.
    This is mathematically optimal with O(log n) complexity.
    
    Phase 1: Exponential search - find range [1, 2, 4, 8, 16, 32, ...]
    Phase 2: Binary search within the range
    
    NOW WITH OPTIMIZATIONS:
    - No duplicate OCR calls (is_text_same returns the text)
    - Uses caching to avoid re-processing frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract reference text (using cache)
    start_text = extract_text_from_frame_cached(video_path, start_frame_index, source_language)
    if start_text is None or start_text == "":
        return -1
    
    print(f"Reference text (frame {start_frame_index}): {start_text[:80]}...")
    
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

def calculate_optimal_batch_config(total_frames: int) -> Tuple[int, int]:
    """
    Calculate optimal batch size and number of workers based on:
    - Total frames in video
    - Available CPU cores
    - Memory considerations
    
    Returns:
        (batch_size, num_workers)
    """
    # Get CPU count (leave 1-2 cores for system)
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, min(cpu_count - 1, 10))  # Cap at 10 workers
    
    # Dynamic batch sizing based on video length
    if total_frames < 500:
        batch_size = 50  # Small videos: smaller batches
    elif total_frames < 3000:
        batch_size = 100  # Medium videos
    elif total_frames < 10000:
        batch_size = 300  # Large videos
    else:
        batch_size = 500  # Very large videos
    
    # Ensure we have enough batches to utilize workers
    num_batches = (total_frames + batch_size - 1) // batch_size
    if num_batches < num_workers:
        # Reduce batch size to create more batches
        batch_size = max(50, total_frames // (num_workers * 2))
    
    print(f"📊 Batch Configuration:")
    print(f"   Total frames: {total_frames}")
    print(f"   Batch size: {batch_size} frames per batch")
    print(f"   Number of batches: {(total_frames + batch_size - 1) // batch_size}")
    print(f"   Workers: {num_workers} parallel threads")
    print(f"   CPU cores available: {cpu_count}")
    
    return batch_size, num_workers


def process_batch_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    batch_id: int,
    source_language: str,
    similarity_threshold: float = 0.85
) -> list:
    """
    Process a batch of frames to find text change points.
    This function runs in a separate thread.
    
    Returns:
        List of (start_frame, end_frame) tuples representing text segments
    """
    print(f"\n🔄 [Batch {batch_id}] Processing frames {start_frame} to {end_frame}")
    
    segments = []
    current_frame = start_frame
    
    while current_frame < end_frame:
        # Find where text changes within this batch
        change_frame = find_text_change_optimal(
            video_path=video_path,
            start_frame_index=current_frame,
            source_language=source_language,
            similarity_threshold=similarity_threshold
        )
        
        if change_frame == -1:
            # No change found, process till end of batch
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
    
    print(f"✅ [Batch {batch_id}] Completed. Found {len(segments)} segment(s)")
    return segments


def find_all_text_segments_parallel(
    video_path: str,
    total_frames: int,
    source_language: str = "english",
    similarity_threshold: float = 0.85
) -> list:
    """
    Find all text change segments in the video using parallel batch processing.
    
    Returns:
        List of (start_frame, end_frame) tuples for each text segment
    """
    batch_size, num_workers = calculate_optimal_batch_config(total_frames)
    
    # Create batch ranges
    batches = []
    for i in range(0, total_frames, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, total_frames)
        batches.append((batch_start, batch_end, len(batches)))
    
    print(f"\n🚀 Starting parallel processing with {num_workers} workers...")
    
    all_segments = []
    
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
                similarity_threshold
            ): (batch_start, batch_end, batch_id)
            for batch_start, batch_end, batch_id in batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_info = future_to_batch[future]
            try:
                segments = future.result()
                all_segments.extend(segments)
                print(f"✓ Batch {batch_info[2]} results collected")
            except Exception as e:
                print(f"❌ Batch {batch_info[2]} failed with error: {e}")
    
    # Sort segments by start frame
    all_segments.sort(key=lambda x: x[0])
    
    # Merge adjacent segments with same text (edge cases between batches)
    merged_segments = merge_adjacent_segments(all_segments, video_path, source_language)
    
    print(f"\n✅ Parallel processing complete!")
    print(f"   Total segments found: {len(merged_segments)}")
    print(f"   Cache size: {len(ocr_cache)} frames")
    
    return merged_segments


def merge_adjacent_segments(segments: list, video_path: str, source_language: str) -> list:
    """
    Merge adjacent segments that have the same text (handles batch boundaries).
    """
    if not segments:
        return []
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last_merged = merged[-1]
        
        # Check if adjacent segments have same text
        if current[0] == last_merged[1]:  # Adjacent segments
            # Get text from both segments
            last_text = extract_text_from_frame_cached(video_path, last_merged[0], source_language)
            current_text = extract_text_from_frame_cached(video_path, current[0], source_language)
            
            similarity = text_similarity(last_text, current_text)
            
            if similarity >= 0.85:  # Same text, merge
                merged[-1] = (last_merged[0], current[1])
            else:
                merged.append(current)
        else:
            merged.append(current)
    
    return merged


def function_overlaying_continuous_legacy(video_path, font_path, font_size, out_path="output/translated.mp4",target_language="English", font_color="black",source_language="english"):
    """
    LEGACY VERSION: Sequential processing (kept for reference)
    Use function_overlaying_continuous() for the new parallel version.
    """
    print(f"Processing video: {video_path}")
    extract_audio(video_path, "input_videos/audio.mp3")
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Open video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    start_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while start_frame < total_frames:
        # Find frame where text changes
        change_frame = find_text_change_optimal(video_path=video_path,start_frame_index=start_frame,source_language=source_language)
        if change_frame == -1:
            change_frame = total_frames  # process till end
        
        # Extract first frame of this segment
        frame = get_frame_at_index(video_path, start_frame)
        if frame is None:
            start_frame = change_frame
            continue
        
        # Extract lines and translate
        lines = extract_lines_with_boxes(frame)
       
        print("Extracted lines:", lines)
        translated_lines = translate_lines(lines,target_language=target_language)
        print(translated_lines)
        # Overlay translated text for all frames in this segment
        for i in range(start_frame, change_frame):
            frame =  get_frame_at_index(video_path, i)
            print(f"Overlaying frame {i}/{total_frames}")
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
        
        print(f"Processed frames {start_frame} to {change_frame - 1}")
        start_frame = change_frame  # Move to next segment
        

    cap.release()
    out.release()
    combine_audio_with_video(silent_video_path=out_path, audio_path="input_videos/audio.mp3", combined_audio_video_path=out_path)
    print("✅ Translation overlay completed for the entire video.")
    
    try:
        if os.path.exists("input_videos/audio.mp3"):
            os.remove("input_videos/audio.mp3")
            print(f"🗑️ Deleted input audio: input_videos/audio.mp3")
        else:
            print(f"⚠️ Input audio not found for deletion:input_videos/audio.mp3")
    except Exception as e:
        print(f"⚠️ Could not delete audio: {e}")


def function_overlaying_continuous(video_path, font_path, font_size, out_path="output/translated.mp4", 
                                   target_language="English", font_color="black", source_language="english",
                                   use_parallel=True):
    """
    NEW VERSION: Parallel batch processing for optimal performance
    
    Steps:
    1. Find all text segments in parallel using batches
    2. Process segments sequentially (overlay and write)
    
    Args:
        use_parallel: If True, use parallel processing; if False, use legacy sequential
    """
    if not use_parallel:
        return function_overlaying_continuous_legacy(
            video_path, font_path, font_size, out_path, target_language, font_color, source_language
        )
    
    print(f"🎬 Processing video: {video_path}")
    print(f"🚀 Using PARALLEL processing mode")
    
    # Extract audio first
    extract_audio(video_path, "input_videos/audio.mp3")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"📹 Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # STEP 1: Find all text segments using parallel processing
    print("\n" + "="*60)
    print("STEP 1: Finding text segments (PARALLEL)")
    print("="*60)
    
    segments = find_all_text_segments_parallel(
        video_path=video_path,
        total_frames=total_frames,
        source_language=source_language,
        similarity_threshold=0.85
    )
    
    if not segments:
        print("⚠️ No text segments found!")
        return
    
    # STEP 2: Process each segment (extract, translate, overlay)
    print("\n" + "="*60)
    print("STEP 2: Overlaying translations")
    print("="*60)
    
    # Open video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    for seg_idx, (start_frame, end_frame) in enumerate(segments, 1):
        print(f"\n📝 Segment {seg_idx}/{len(segments)}: frames {start_frame}-{end_frame}")
        
        # Extract first frame of this segment
        frame = get_frame_at_index(video_path, start_frame)
        if frame is None:
            print(f"⚠️ Could not read frame {start_frame}, skipping segment")
            continue
        
        # Extract lines and translate
        lines = extract_lines_with_boxes(frame)
        print(f"   Extracted {len(lines)} text lines")
        
        translated_lines = translate_lines(lines, target_language=target_language)
        print(f"   Translated to {target_language}")
        
        # Overlay translated text for all frames in this segment
        frames_in_segment = end_frame - start_frame
        for i in range(start_frame, end_frame):
            frame = get_frame_at_index(video_path, i)
            
            if i % 10 == 0 or i == start_frame:  # Progress every 10 frames
                progress = ((i - start_frame + 1) / frames_in_segment) * 100
                print(f"   Overlaying frame {i}/{end_frame} ({progress:.1f}%)")
            
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
        
        print(f"   ✅ Segment {seg_idx} complete")
    
    out.release()
    
    # STEP 3: Combine with audio
    print("\n" + "="*60)
    print("STEP 3: Adding audio")
    print("="*60)
    combine_audio_with_video(
        silent_video_path=out_path, 
        audio_path="input_videos/audio.mp3", 
        combined_audio_video_path=out_path
    )
    
    # Cleanup
    try:
        if os.path.exists("input_videos/audio.mp3"):
            os.remove("input_videos/audio.mp3")
            print(f"🗑️ Cleaned up temporary audio file")
    except Exception as e:
        print(f"⚠️ Could not delete audio: {e}")
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Total frames: {total_frames}")
    print(f"   OCR cache size: {len(ocr_cache)} frames")
    print(f"   Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay translations on video frames with PARALLEL processing support."
    )
    parser.add_argument("--video", dest="video_path", required=True, help="Path to input video")
    parser.add_argument("--font", dest="font_path", default=None, help="Path to TTF font (optional)")
    parser.add_argument("--fontSize", dest="font_size", default="35", help="Font size (int)")
    parser.add_argument("--out", dest="out_path", default="output/translated.mp4", help="Output video path")
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
