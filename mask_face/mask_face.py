import math
import time

import cv2
import numpy as np
from PIL import Image


class FpsWatcher:
    """fps"""

    def __init__(self):
        self.t = time.time()

    def __call__(self):
        fps = 1.0 / (time.time() - self.t)
        self.t = time.time()
        return fps


class GifStreamer:
    def __init__(self, path, cut_color=(0, 0, 0)):
        self.split2frame(path)
        self.frame_list = [add_alphach(frame, cut_color) for frame in self.frame_list]
        self.frame_list = [cut_alphablank(frame) for frame in self.frame_list]

    def split2frame(self, path):
        self.frame_list = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.frame_list.append(frame)
            else:
                cap.release()
        self.idx = 0

    def __len__(self):
        return len(self.frame_list)

    def _read(self, idx):
        return self.frame_list[idx]

    def __call__(self):
        frame = self._read(self.idx)
        self.idx = (self.idx + 1) % len(self)
        return frame


def expand_bbox(_bbox, expand_rate=1.2):
    bbox = []
    w_margin = int((expand_rate - 1) * (np.max(_bbox.T[0]) - np.min(_bbox.T[0])))
    h_margin = int((expand_rate - 1) * (np.max(_bbox.T[1]) - np.min(_bbox.T[1])))
    for i, bb in enumerate(_bbox, 1):
        sign_x = 1 if i % 2 == 0 else -1
        sign_y = -1 if i <= 2 else 1
        bbox.append(bb + np.array([sign_x * w_margin, sign_y * h_margin]))
    return np.asarray(bbox)


def cut_alphablank(img):
    assert img.shape[2] == 4, "image must have alpha-channel"
    y_indices, x_indices = np.where(img[:, :, -1] > 0)
    left = np.min(x_indices)
    top = np.min(y_indices)
    right = np.max(x_indices)
    bottom = np.max(y_indices)
    return img[top:bottom, left:right].copy()


def add_alphach(img, color=(0, 0, 0), upper=10, lower=10):
    color = np.array(color)
    alpha_img = img.copy()
    alpha_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = cv2.inRange(img, color - lower, color + upper)
    alpha_img[:, :, -1] = 255 - mask.copy()
    return alpha_img


def paste_facemask(
    base_image, paste_image, landmarks, size=(720, 1280), expand_rate=1.2
):
    assert paste_image.shape[2] == 4, "paste image must have alpha-channel"
    # calc bbox
    x_range = [np.min(landmarks.T[0]), np.max(landmarks.T[0])]
    y_range = [np.min(landmarks.T[1]), np.max(landmarks.T[1])]
    center = np.mean([x_range, y_range], axis=1)
    bbox = np.asarray([[x, y] for y in y_range for x in x_range])
    bbox = expand_bbox(bbox, expand_rate)

    # calc Rotation Matrix
    right_idx, left_index = 454, 234
    h_vec = landmarks[right_idx] - landmarks[left_index]
    theta = math.atan2(h_vec[1], h_vec[0])
    Rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # rot!
    rot_bbox = bbox - center
    rot_bbox = np.dot(Rot, rot_bbox.T).T + center
    rot_bbox = rot_bbox.astype(np.int64)

    # make mask
    paste_image = cut_alphablank(paste_image)
    ph, pw = paste_image.shape[:2]
    mask = np.zeros((size[0], size[1], 4))
    mask[0:ph, 0:pw] = paste_image[:, :, :].copy()
    pts1 = np.float32([[x, y] for y in [0, ph] for x in [0, pw]])
    pts2 = rot_bbox.astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    mask = cv2.warpPerspective(mask, M, (size[1], size[0]))

    img = Image.fromarray(base_image)
    alpha = Image.fromarray(mask.astype(np.uint8))
    img.paste(alpha, mask=alpha)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB)
    return img
