import cv2
import os




def extract_frame_from_video():
    # Path to your input video
    video_path = os.path.join( 'input_videos', 'test2.mp4')
    print(f"Video path: {video_path}")
    # Path to save the extracted frame
    output_path = os.path.join( 'output_images', 'frame_0.jpg')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 7)

    # Read the frame
    ret, frame = cap.read()


    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Frame saved to {output_path}")
    else:
        print("Could not read frame from video.")

    cap.release()


extract_frame_from_video()