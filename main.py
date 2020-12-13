import argparse
import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mask_face import FpsWatcher, GifStreamer, cut_alphablank, paste_facemask


def get_landmarks(mp_landmarks, size=(720, 1280)):
    h, w = size
    landmarks = np.asarray(
        [[l.x * w, l.y * h] for l in mp_landmarks],
        dtype=np.int32,
    )
    return landmarks


def main(cfg):

    # init
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    drawFace = False
    drawHand = True
    drawPose = False

    holistic = mp_holistic.Holistic(
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        upper_body_only=cfg.upper_body_only,
    )
    cap = cv2.VideoCapture(cfg.device)

    fps = FpsWatcher()
    path = Path(cfg.img_path)
    assert path.is_file(), "file is not found"
    img_mode = "gif" if path.suffix == ".gif" else "png"
    if img_mode == "gif":
        paste_gif = GifStreamer(str(path))
    else:
        paste_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # draw
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks is not None:
            landmarks = get_landmarks(results.face_landmarks.landmark, image.shape[:2])
            paste_image = paste_gif() if img_mode == "gif" else paste_image
            image = paste_facemask(
                image, paste_image, landmarks, expand_rate=1.5, size=image.shape[:2]
            )

        cv2.putText(
            image,
            f"FPS{int(fps())}",
            (0, 50),
            cv2.FONT_HERSHEY_PLAIN,
            4,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if drawFace:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS
            )
        if drawHand:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
        if drawPose and not cfg.upper_body_only:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )

        cv2.imshow("MediaPipe Holistic", image)

        key = cv2.waitKey(5)
        if key & 0xFF == 27:
            break
        elif key == ord("f"):
            drawFace = not drawFace
        elif key == ord("h"):
            drawHand = not drawHand
        elif key == ord("p"):
            drawPose = not drawPose

    holistic.close()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="masks/mask.gif")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--upper_body_only", type=bool, default=False)
    args = parser.parse_args()
    try:
        main(args)
    except AssertionError:
        traceback.print_exc()
