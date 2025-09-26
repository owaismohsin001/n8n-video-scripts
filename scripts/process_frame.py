# import cv2
# import os

# def extract_frame_from_video():
#     # Path to your input video
#     video_path = os.path.join( 'input_videos', 'test2.mp4')
#     print(f"Video path: {video_path}")
#     # Path to save the extracted frame
#     output_path = os.path.join( 'output_images', 'frame_0.jpg')

#     # Open the video
#     cap = cv2.VideoCapture(video_path)

#     cap.set(cv2.CAP_PROP_POS_FRAMES, 7)

#     # Read the frame
#     ret, frame = cap.read()


#     if ret:
#         cv2.imwrite(output_path, frame)
#         print(f"Frame saved to {output_path}")
#     else:
#         print("Could not read frame from video.")

#     cap.release()


# extract_frame_from_video()


import cv2
import os
import time

def extract_frame_from_video(video_filename='test2.mp4', frame_number=7, output_dir='output_images'):
    """
    Extract a single frame from a video and save it with a unique name.
    Returns the path of the saved image.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build paths
    video_path = os.path.join('input_videos', video_filename)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)          # frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames
    duration = frame_count / fps    
    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds")
    # Go to the specific frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")

    # Generate a unique name: frame_<timestamp>_<frame_number>.jpg
    unique_name = f"frame_{frame_number}_{int(time.time())}.jpg"
    output_path = os.path.join(output_dir, unique_name)

    # Save the frame
    cv2.imwrite(output_path, frame)
    cap.release()

    print(f"Frame saved to {output_path}")
    return output_path
