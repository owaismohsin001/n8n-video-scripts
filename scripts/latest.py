# First, install easyocr if not already installed
# Run this in a cell: !pip install easyocr

import cv2
import easyocr
from difflib import SequenceMatcher
from overlay_utils import overlay_translated_lines_on_frame  
from helpers import fuzzy_get
from translate_utils import translate_lines
from ocr_utils import extract_lines_with_boxes  
from optimal_latest import  find_text_change_bidirectional
# Initialize reader once globally to avoid reloading models
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

def find_text_change_frame(video_path, start_frame_index, similarity_threshold=0.85):
    """
    Find the frame index where text changes from the starting frame.
    
    Args:
        video_path: Path to the video file
        start_frame_index: Starting frame index to compare against
        similarity_threshold: Threshold for considering text as "similar" (0-1)
    
    Returns:
        Frame index where text changes, or -1 if no change found
    """
    # Open video to get total frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract text from starting frame
    start_frame = get_frame_at_index(video_path, start_frame_index)
    if start_frame is None:
        print(f"Error: Could not read frame {start_frame_index}")
        return -1
    
    start_text = extract_text_easyocr(start_frame)
    print(f"Start frame ({start_frame_index}) text: {start_text[:100]}...")
    
    # Initialize search parameters
    current_index = start_frame_index
    step_size = 20  # Initial large step
    last_same_index = start_frame_index
    
    # Phase 1: Jump by 20 frames until text differs
    print("\n=== Phase 1: Jumping by 20 frames ===")
    while current_index + step_size < total_frames:
        current_index += step_size
        frame = get_frame_at_index(video_path, current_index)
        
        if frame is None:
            break
            
        current_text = extract_text_easyocr(frame)
        similarity = text_similarity(start_text, current_text)
        
        print(f"Frame {current_index}: Similarity = {similarity:.2f}")
        
        if similarity < similarity_threshold:
            # Text changed! Move to phase 2
            print(f"Text differs at frame {current_index}")
            break
        else:
            last_same_index = current_index
    
    # Phase 2: Move back 5 frames and check
    print("\n=== Phase 2: Moving back 5 frames ===")
    current_index = last_same_index + 5
    
    while current_index < total_frames:
        frame = get_frame_at_index(video_path, current_index)
        
        if frame is None:
            break
            
        current_text = extract_text_easyocr(frame)
        similarity = text_similarity(start_text, current_text)
        
        print(f"Frame {current_index}: Similarity = {similarity:.2f}")
        
        if similarity < similarity_threshold:
            # Text changed! Move to phase 3
            print(f"Text differs at frame {current_index}")
            break
        else:
            last_same_index = current_index
            current_index += 5
    
    # Phase 3: Move by 1 frame to find exact change point
    print("\n=== Phase 3: Fine-tuning by 1 frame ===")
    current_index = last_same_index + 1
    
    while current_index < total_frames:
        frame = get_frame_at_index(video_path, current_index)
        
        if frame is None:
            break
            
        current_text = extract_text_easyocr(frame)
        similarity = text_similarity(start_text, current_text)
        
        print(f"Frame {current_index}: Similarity = {similarity:.2f}")
        
        if similarity < similarity_threshold:
            print(f"\n✓ Text change detected at frame {current_index}")
            print(f"Original text: {start_text[:100]}...")
            print(f"New text: {current_text[:100]}...")
            return current_index
        
        last_same_index = current_index
        current_index += 1
    
    print(f"\nNo text change found. Last checked frame: {last_same_index}")
    return -1

cap = cv2.VideoCapture("input_videos/test_cut.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_latest/translated.mp4", fourcc, fps, (width, height))
# def function_overlaying():
#     start_frame = 1
#     video_path = "input_videos/test2.mp4"
#     change_frame = find_text_change_frame(video_path, start_frame)
#     frame = get_frame_at_index(video_path, i)
#     lines= extract_lines_with_boxes(frame)
#     translated_lines = translate_lines(lines, target_language="English")
#     for i in range(start_frame, change_frame):
# 	    frame = get_frame_at_index(video_path, i)
# 		frame_with_overlay = overlay_translated_lines_on_frame( frame, translated_lines, font_path="fonts/NotoSans-Regular.ttf",font_size=45)
# 	    out.write(frame_with_overlay)
#     if change_frame != -1:
#         print(f"\nResult: Text remains similar from frame {start_frame} to {change_frame - 1}")
#         print(f"Total frames with same text: {change_frame - start_frame}")
#     else:
#         print("\nNo change in text detected until end of video")

# function_overlaying()


# def function_overlaying():
#     start_frame = 1
#     video_path = "input_videos/test2.mp4"

#     # Find the frame where text changes
#     change_frame = find_text_change_frame(video_path, start_frame)
#     # Extract the first frame to get the lines
#     frame = get_frame_at_index(video_path, start_frame)
#     lines = extract_lines_with_boxes(frame)
#     translated_lines = translate_lines(lines, target_language="English")
    
#     # Open video for writing output
#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter("output/translated.mp4", fourcc, fps, (width, height))
    
#     # Overlay translated text on frames
#     for i in range(start_frame, change_frame):
#         frame = get_frame_at_index(video_path, i)
#         if frame is None:
#             continue
#         frame_with_overlay = overlay_translated_lines_on_frame(
#             frame,
#             translated_lines,
#             font_path="fonts/NotoSans-Regular.ttf",
#             font_size=45
#         )
#         out.write(frame_with_overlay)
    
#     cap.release()
#     out.release()
    
#     if change_frame != -1:
#         print(f"\nResult: Text remains similar from frame {start_frame} to {change_frame - 1}")
#         print(f"Total frames with same text: {change_frame - start_frame}")
#     else:
#         print("\nNo change in text detected until end of video")
# function_overlaying()


def function_overlaying_continuous():
    video_path = "input_videos/test_cut.mp4"
    
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Open video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter("optimal_output/translated.mp4", fourcc, fps, (width, height))
    
    start_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while start_frame < total_frames:
        # Find frame where text changes
        change_frame = find_text_change_bidirectional(video_path, start_frame)
        if change_frame == -1:
            change_frame = total_frames  # process till end
        
        # Extract first frame of this segment
        frame = get_frame_at_index(video_path, start_frame)
        if frame is None:
            start_frame = change_frame
            continue
        
        # Extract lines and translate
        lines = extract_lines_with_boxes(frame)
        translated_lines = translate_lines(lines, target_language="English")
        
        # Overlay translated text for all frames in this segment
        for i in range(start_frame, change_frame):
            print()
            frame =  get_frame_at_index(video_path, i)
            if frame is None:
                continue
            frame_with_overlay = overlay_translated_lines_on_frame(
                frame,
                translated_lines,
                font_path="fonts/NotoSans-Regular.ttf",
                font_size=35
            )
            out.write(frame_with_overlay)
        
        print(f"Processed frames {start_frame} to {change_frame - 1}")
        start_frame = change_frame  # Move to next segment
    
    cap.release()
    out.release()
    print("✅ Translation overlay completed for the entire video.")

function_overlaying_continuous()
