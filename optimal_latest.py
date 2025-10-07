# First, install easyocr if not already installed
# Run this in a cell: !pip install easyocr

import os
import cv2
import easyocr
from difflib import SequenceMatcher # returns np array frame
# Initialize reader once globally to avoid reloading models
from overlay_utils import overlay_translated_lines_on_frame  
from translate_utils import translate_lines
from ocr_utils import extract_lines_with_boxes  
import argparse



reader = None

def get_reader():
    """Get or create EasyOCR reader instance"""
    global reader
    if reader is None:
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)  # CPU mode
    return reader

def extract_text_easyocr(image):
    """Extract text from image using EasyOCR"""
    ocr_reader = get_reader()
    results = ocr_reader.readtext(image)
    return " ".join([res[1] for res in results])

def text_similarity(text1, text2):
    """Calculate similarity between two texts (0 to 1)"""
    return SequenceMatcher(None, text1, text2).ratio()

def get_frame_at_index(video_path, frame_index):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        cap.release()
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_index < 0 or frame_index >= total_frames:
        print(f"Error: Frame index {frame_index} out of range (0-{total_frames-1})")
        cap.release()
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_index}")
        return None
    
    return frame

def is_text_same(video_path, frame_index, reference_text, similarity_threshold):
    """Check if text at frame_index is similar to reference_text"""
    frame = get_frame_at_index(video_path, frame_index)
    if frame is None:
        return None
    
    current_text = extract_text_easyocr(frame)
    similarity = text_similarity(reference_text, current_text)
    return similarity >= similarity_threshold


# ============================================================================
# METHOD 1: YOUR BIDIRECTIONAL APPROACH [500, 100, 20, 5, 1]
# ============================================================================
def find_text_change_bidirectional(video_path, start_frame_index, similarity_threshold=0.85):
    """
    Find text change using bidirectional search: forward until different, 
    back until same, repeat with smaller steps.
    
    Steps: [500, 100, 20, 5, 1]
    - Move 500 forward until DIFFERENT
    - Move 100 back until SAME
    - Move 20 forward until DIFFERENT
    - Move 5 back until SAME
    - Move 1 forward until DIFFERENT (exact frame)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract reference text
    start_frame = get_frame_at_index(video_path, start_frame_index)
    if start_frame is None:
        return -1
    
    start_text = extract_text_easyocr(start_frame)
    print(f"Reference text (frame {start_frame_index}): {start_text[:80]}...")
    
    # Bidirectional search steps: (step_size, direction)
    # direction: 1 = forward (find DIFFERENT), -1 = backward (find SAME)
    search_phases = [
        (500, 1, "DIFFERENT"),   # Move forward 500 until different
        (100, -1, "SAME"),       # Move back 100 until same
        (20, 1, "DIFFERENT"),    # Move forward 20 until different
        (5, -1, "SAME"),         # Move back 5 until same
        (1, 1, "DIFFERENT")      # Move forward 1 until different (exact)
    ]
    
    current_index = start_frame_index
    frame_checks = 0
    
    for phase_num, (step_size, direction, target_state) in enumerate(search_phases, 1):
        print(f"\n=== Phase {phase_num}: Step {step_size} {'FORWARD' if direction == 1 else 'BACKWARD'} (find {target_state}) ===")
        
        while True:
            next_index = current_index + (step_size * direction)
            
            # Boundary checks
            if next_index < start_frame_index or next_index >= total_frames:
                # print(f"Reached boundary at frame {current_index}")
                break
            
            # Check frame
            result = is_text_same(video_path, next_index, start_text, similarity_threshold)
            frame_checks += 1
            
            if result is None:
                break
            
            is_same = result
            similarity = text_similarity(start_text, extract_text_easyocr(get_frame_at_index(video_path, next_index)))
            
            print(f"Frame {next_index}: {'SAME' if is_same else 'DIFFERENT'} (similarity={similarity:.3f})")
            
            # Check if we found what we're looking for
            if target_state == "DIFFERENT" and not is_same:
                # Found different text
                if step_size == 1:
                    # This is the exact frame!
                    print(f"\n✓ Exact text change frame: {next_index}")
                    print(f"Total frames checked: {frame_checks}")
                    return next_index
                else:
                    # Move to next phase
                    current_index = next_index
                    break
            elif target_state == "SAME" and is_same:
                # Found same text (backtracked successfully)
                current_index = next_index
                break
            else:
                # Keep searching
                current_index = next_index
    
    print(f"\nNo text change found. Total frames checked: {frame_checks}")
    return -1


# ============================================================================
# METHOD 2: EXPONENTIAL SEARCH + BINARY SEARCH (OPTIMAL)
# ============================================================================
def find_text_change_optimal(video_path, start_frame_index, similarity_threshold=0.85):
    """
    Find text change using exponential search + binary search.
    This is mathematically optimal with O(log n) complexity.
    
    Phase 1: Exponential search - find range [1, 2, 4, 8, 16, 32, ...]
    Phase 2: Binary search within the range
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract reference text
    start_frame = get_frame_at_index(video_path, start_frame_index)
    if start_frame is None:
        return -1
    
    start_text = extract_text_easyocr(start_frame)
    print(f"Reference text (frame {start_frame_index}): {start_text[:80]}...")
    
    frame_checks = 0
    
    # Phase 1: Exponential search to find upper bound
    print("\n=== Phase 1: Exponential Search ===")
    step = 1
    current_index = start_frame_index + step
    
    while current_index < total_frames:
        result = is_text_same(video_path, current_index, start_text, similarity_threshold)
        frame_checks += 1
        
        if result is None:
            break
        
        similarity = text_similarity(start_text, extract_text_easyocr(get_frame_at_index(video_path, current_index)))
        print(f"Frame {current_index} (step={step}): {'SAME' if result else 'DIFFERENT'} (similarity={similarity:.3f})")
        
        if not result:  # Found different text
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
        result = is_text_same(video_path, mid, start_text, similarity_threshold)
        frame_checks += 1
        
        if result is None:
            break
        
        similarity = text_similarity(start_text, extract_text_easyocr(get_frame_at_index(video_path, mid)))
        print(f"Frame {mid}: {'SAME' if result else 'DIFFERENT'} (similarity={similarity:.3f})")
        
        if result:  # Still same text
            left = mid + 1
        else:  # Different text
            right = mid
    
    print(f"\n✓ Exact text change frame: {left}")
    print(f"Total frames checked: {frame_checks}")
    return left




def function_overlaying_continuous(video_path, font_path, font_size, out_path="output/translated.mp4",target_language="English"):
    # video_path = "input_videos/test_cut.mp4"
    print(f"Processing video: {video_path}")
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Open video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter("output/translated.mp4", fourcc, fps, (width, height))
    
    start_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while start_frame < total_frames:
        # Find frame where text changes
        change_frame = find_text_change_optimal(video_path=video_path,start_frame_index=start_frame)
        if change_frame == -1:
            change_frame = total_frames  # process till end
        
        # Extract first frame of this segment
        frame = get_frame_at_index(video_path, start_frame)
        if frame is None:
            start_frame = change_frame
            continue
        
        # Extract lines and translate
        lines = extract_lines_with_boxes(frame)
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
                font_size=font_size
            )
            out.write(frame_with_overlay)
        
        print(f"Processed frames {start_frame} to {change_frame - 1}")
        start_frame = change_frame  # Move to next segment
    
    cap.release()
    out.release()
    print("✅ Translation overlay completed for the entire video.")
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑️ Deleted input video: {video_path}")
        else:
            print(f"⚠️ Input video not found for deletion: {video_path}")
    except Exception as e:
        print(f"⚠️ Could not delete input video: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay translations on video frames.")
    parser.add_argument("--video", dest="video_path", required=True, help="Path to input video")
    parser.add_argument("--font", dest="font_path", default=None, help="Path to TTF font (optional)")
    parser.add_argument("--fontSize", dest="font_size", default="35", help="Font size (int)")
    parser.add_argument("--out", dest="out_path", default="output/translated.mp4", help="Output video path")
    parser.add_argument("--targetLang", dest="target_language", default="ch_sim", help="Target language for translation")
    args = parser.parse_args()
    function_overlaying_continuous(
        video_path=args.video_path,
        font_path=args.font_path,
        font_size=int(args.font_size),
        out_path=args.out_path,
        target_language=args.target_language
    )
#   "en", "de", "es