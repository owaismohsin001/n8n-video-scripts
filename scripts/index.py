# import cv2
# import os
# import pytesseract
# from dotenv import load_dotenv
# import cv2
# from pytesseract import Output
# import pandas as pd
# from overlay_utils import overlay_translated_lines_on_frame
# from translate_utils import translate_lines
# from process_frame import extract_frame_from_video
# from ocr_utils import extract_lines_with_boxes
# import difflib
# load_dotenv()

# def get_text_only(lines):
#     """Collapse whitespace and extract just the text part from OCR lines."""
#     return [' '.join(text.split()) for text, _ in lines]

# def is_similar(texts1, texts2, threshold=0.95):
#     print(texts1, texts2, threshold)
#     """Return True if the joined texts are at least threshold similar."""
#     joined1 = ' '.join(texts1)
#     joined2 = ' '.join(texts2)
#     ratio = difflib.SequenceMatcher(None, joined1, joined2).ratio()
#     print(ratio >= threshold)
#     return ratio >= threshold


# video_path = "input_videos/test2.mp4"
# output_path = "output_videos/test2_translated.mp4"

# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # define video writer with same FPS/resolution
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# os.makedirs("output_videos", exist_ok=True)

# skip = 30  # process every 30th frame
# frame_number = 4
# last_overlay = None

# # step 1

# # while True:
# #     last_text_extracted = []
# #     ret, frame = cap.read()
# #     image_path=extract_frame_from_video(video_filename='test2.mp4', frame_number=frame_number, output_dir='output_images')
# #     if not ret:
# #         break
# #     if frame_number % skip == 0 or frame_number == 4:
# #         # Heavy OCR/translation only every 30th frame
# #         # image_path=extract_frame_from_video(video_filename='test2.mp4', frame_number=frame_number, output_dir='output_images')
# #         lines = extract_lines_with_boxes(image_path)
# #         translated_lines = translate_lines(lines, target_language="English")
# #         last_overlay = translated_lines 
# #         # cache result
# #     # Always draw last overlay (even for skipped frames)
# #     print(frame_number, last_overlay)
# #     result_img = overlay_translated_lines(image_path, last_overlay, font_path="fonts/NotoSans-Regular.ttf", font_size=45)
# #     result_img.save(image_path)

# #     # out.write(frame_with_overlay)  # save to video
# #     frame_number += 1

# # cap.release()
# # out.release()

# # step 2

# # --- your loop ---
# # last_text_extracted = None   # store just the text portions
# # last_overlay = None

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     image_path = extract_frame_from_video(
# #         video_filename='test2.mp4',
# #         frame_number=frame_number,
# #         output_dir='output_images'
# #     )

# #     if frame_number % skip == 0 or frame_number == 4:
# #         # OCR only every skip frames
# #         lines = extract_lines_with_boxes(image_path)
# #         text_only = get_text_only(lines)

# #         if (last_text_extracted is None) or (not is_similar(text_only, last_text_extracted)):
# #             # only translate if new text detected (not similar enough)
# #             translated_lines = translate_lines(lines, target_language="English")
# #             last_overlay = translated_lines
# #             last_text_extracted = text_only
# #         # else: keep last_overlay as-is

# #     # Always draw overlay
# #     if last_overlay is not None:
# #         result_img = overlay_translated_lines(
# #             image_path,
# #             last_overlay,
# #             font_path="fonts/NotoSans-Regular.ttf",
# #             font_size=45
# #         )
# #         result_img.save(image_path)

# #     frame_number += 1

# # cap.release()
# # out.release()


# #step 3

# last_text_extracted = None
# last_overlay = None
# frame_number = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_number % skip == 0 or frame_number == 4:
#         lines = extract_lines_with_boxes(frame)
#         text_only = get_text_only(lines)

#         if last_text_extracted is None or not is_similar(text_only, last_text_extracted):
#             translated_lines = translate_lines(lines, target_language="English")
#             last_overlay = translated_lines
#             last_text_extracted = text_only
#         # else: reuse last_overlay

#     if last_overlay is not None:
#         frame_with_overlay = overlay_translated_lines_on_frame(
#             frame,
#             last_overlay,
#             font_path="fonts/NotoSans-Regular.ttf",
#             font_size=45
#         )
#         out.write(frame_with_overlay)
#     else:
#         out.write(frame)

#     frame_number += 1

# cap.release()
# out.release()




import cv2
import os
import pytesseract
from dotenv import load_dotenv
from pytesseract import Output
import pandas as pd
from overlay_utils import overlay_translated_lines_on_frame  # returns np array frame
from translate_utils import translate_lines
from ocr_utils import extract_lines_with_boxes  # modified to accept np array
import difflib

load_dotenv()


def get_text_only(lines):
    """Collapse whitespace and extract just the text part from OCR lines."""
    return [' '.join(text.split()) for text, _ in lines]


def is_similar(texts1, texts2, threshold=0.80):
    """Return True if the joined texts are at least threshold similar."""
    joined1 = ' '.join(texts1)
    joined2 = ' '.join(texts2)
    # print(texts1, texts2, threshold)
    ratio = difflib.SequenceMatcher(None, joined1, joined2).ratio()
    # print(ratio,ratio >= threshold)
    return ratio >= threshold


video_path = "input_videos/test2.mp4"
output_path = "output_videos/test2_translated.mp4"

# make sure output folder exists BEFORE writing
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

skip = 30  # process every 30th frame
# frame_number = 9000
# last_text_extracted = None
# last_overlay = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # do OCR every skip frames
#     if frame_number % skip == 0 or frame_number == 7200:
#         lines = extract_lines_with_boxes(frame)  # <-- now frame, not path
#         print(lines,"lines")
#         text_only = get_text_only(lines)

#         if last_text_extracted is None or not is_similar(text_only, last_text_extracted):
#             translated_lines = translate_lines(lines, target_language="English")
#             print(translated_lines)
#             last_overlay = translated_lines
#             last_text_extracted = text_only
#         # else reuse last_overlay

#     # Always draw overlay if we have it
#     if last_overlay is not None:
#         # overlay_translated_lines_on_frame must return np.ndarray (BGR)
#         frame_with_overlay = overlay_translated_lines_on_frame(
#             frame,
#             last_overlay,
#             font_path="fonts/NotoSans-Regular.ttf",
#             font_size=45
#         )
#         out.write(frame_with_overlay)
#     else:
#         out.write(frame)

#     frame_number += 1

# cap.release()
# out.release()

# print(f"Done. Translated video saved to: {output_path}")

#step 4 not handled grace frames
# last_text_extracted = None
# last_overlay = None
# frame_number = 8000

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # do OCR every skip frames
#     if frame_number % skip == 0 or frame_number == 9200:
#         lines = extract_lines_with_boxes(frame)  # frame not path
#         print(lines, "lines")

#         if lines:  # only process when OCR saw something
#             text_only = get_text_only(lines)

#             if (last_text_extracted is None) or (not is_similar(text_only, last_text_extracted)):
#                 translated_lines = translate_lines(lines, target_language="English")
#                 print(translated_lines)
#                 last_overlay = translated_lines
#                 last_text_extracted = text_only
#             # else reuse last_overlay silently

#         # if lines == [] just keep last_overlay/last_text_extracted as they are

#     # Always draw overlay if we have it
#     if last_overlay is not None:
#         # overlay_translated_lines_on_frame must return np.ndarray (BGR)
#         frame_with_overlay = overlay_translated_lines_on_frame(
#             frame,
#             last_overlay,
#             font_path="fonts/NotoSans-Regular.ttf",
#             font_size=45
#         )
#         out.write(frame_with_overlay)
#     else:
#         out.write(frame)

#     frame_number += 1

# cap.release()
# out.release()
# print(f"Done. Translated video saved to: {output_path}")

# step 5- grace frames handled
# last_text_extracted = None
# last_overlay = None
# frame_number = 10000
# grace_counter = 0
# GRACE_FRAMES = 3  # keep overlay for up to 3 frames if OCR fails

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # OCR every `skip` frames
#     if frame_number % skip == 0 or frame_number == 10000:
#         lines = extract_lines_with_boxes(frame)  # frame input
#         print(lines, "lines")

#         if lines:
#             text_only = get_text_only(lines)
#             if last_text_extracted is None or not is_similar(text_only, last_text_extracted):
#                 translated_lines = translate_lines(lines, target_language="English")
#                 last_overlay = translated_lines
#                 last_text_extracted = text_only
#             grace_counter = 0  # reset grace counter
#         else:
#             # No text detected → increment grace counter
#             grace_counter += 1
#             if grace_counter > GRACE_FRAMES:
#                 last_overlay = None
#                 last_text_extracted = None

#     # Draw overlay if available
#     if last_overlay is not None:
#         frame_with_overlay = overlay_translated_lines_on_frame(
#             frame,
#             last_overlay,
#             font_path="fonts/NotoSans-Regular.ttf",
#             font_size=45
#         )
#         out.write(frame_with_overlay)
#     else:
#         out.write(frame)

#     frame_number += 1

# cap.release()
# out.release()




skip = 30  # process every 30th frame
frame_number = 9990
last_text_extracted = None
last_overlay = None
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1 
    print(current_frame_index,"current_frame_index")
    # do OCR every skip frames
    if current_frame_index % skip == 0 or current_frame_index == 9990:
        lines = extract_lines_with_boxes(frame)  # <-- now frame, not path
        # print(lines,"lines")
        text_only = get_text_only(lines)
        if lines:
            if last_text_extracted is None or not is_similar(text_only, last_text_extracted):
                translated_lines = translate_lines(lines, target_language="English")
                # print(translated_lines,"translated_lines")
                print(last_overlay,"last_overlay before update")
                print(last_text_extracted,"last_text_extracted before update")
                last_overlay = translated_lines
                last_text_extracted = text_only
                print(last_overlay,"last_overlay after update")
                print(last_text_extracted,"last_text_extracted after update")
                frame_with_overlay = overlay_translated_lines_on_frame(
                frame,
                last_overlay,
                font_path="fonts/NotoSans-Regular.ttf",
                font_size=45
                )
                debug_path = os.path.join("empty_frames", f"frame_{frame_number}_new_overlay.jpg")
                cv2.imwrite(debug_path, frame_with_overlay)
                out.write(frame_with_overlay)
            # else reuse last_overlay
            else:
                frame_with_overlay = overlay_translated_lines_on_frame(
                frame,
                last_overlay,
                font_path="fonts/NotoSans-Regular.ttf",
                font_size=45
                )
                out.write(frame_with_overlay)
        else:
            last_overlay=None
            last_text_extracted=None
            # frame_without_overlay = overlay_translated_lines_on_frame(
            #     frame,
            #     last_overlay,
            #     font_path="fonts/NotoSans-Regular.ttf",
            #     font_size=45
            # )
            # out.write(frame_without_overlay)

            out.write(frame)
    else:
        if last_overlay is not None:
            frame_with_overlay = overlay_translated_lines_on_frame(
                frame,
                last_overlay,
                font_path="fonts/NotoSans-Regular.ttf",
                font_size=45
            )
            # print(frame_number,"frame_number")
            # # if(frame_number==10649):
            #       debug_path = os.path.join("empty_frames", f"frame_{frame_number}_new_overlay.jpg")
            #       cv2.imwrite(debug_path, frame_with_overlay)
            out.write(frame_with_overlay)
    # frame_number += 1

cap.release()
out.release()

print(f"Done. Translated video saved to: {output_path}")