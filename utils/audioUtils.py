import os
import subprocess

def extract_audio(video_path: str, output_audio_path: str):
    """
    Extracts audio from a video using FFmpeg and saves it to the given path.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        # FFmpeg command to extract audio
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-q:a", "0", "-map", "a",
            output_audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted successfully → {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print("Error extracting audio:", e.stderr.decode())
    except Exception as e:
        print("Unexpected error:", e)


def combine_audio_with_video(silent_video_path: str, audio_path: str, combined_audio_video_path:str):
    """
    Replaces a silent video with a version that includes the provided audio.
    The output overwrites the original file.
    """
    print(f"Combining audio {audio_path} with video {silent_video_path} into {combined_audio_video_path}")
    try:
        output_path = combined_audio_video_path  # overwrite with same name
        temp_output = "/home/node/.n8n/.n8n/binaryData/temp_combined.mp4"
        # temp_output="temp_combined.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-i", silent_video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
            temp_output
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Replace original video with new one
        os.replace(temp_output, output_path)
        print(f"Video updated with audio → {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error combining audio and video:", e.stderr.decode())
    except Exception as e:
        print("Unexpected error:", e)

def remove_audio_from_video(video_path: str, output_video_path: str):
    """
    Removes audio from a video and saves a new silent video.
    Does not modify the original video.
    """
    try:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-an",  # removes audio
            "-c:v", "copy",
            output_video_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Audio removed. Silent video saved → {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("❌ Error removing audio:", e.stderr.decode())



# extract_audio("/content/test_cut.mp4", "/content/audio.mp3")
# combine_audio_with_video("/content/silent_video.mp4", "/content/audio.mp3", "/content/combined_audio_video.mp4")
# remove_audio_from_video("/content/test_cut.mp4", "/content/silent_video.mp4")