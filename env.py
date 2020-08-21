import cv2
import numpy as np


class Env(object):
    def __init__(self):
        self._index = 0
        self.capacity = 1e3
        self._img = []
        self.n_imgs = 4
        self.stack = [None] * self.n_imgs

    def load_video(self, video_path, data_path):
        vid = cv2.VideoCapture(video_path)
        for img in vid:
            self._img.append(img)

        self.stack = [self._img[0]] * self.n_imgs

    def __iter__(self):
        self.stack.pop(0)
        self._index += 1
        if self._index >= self.capacity - self.n_imgs:
            self._index = 0

        img = self._img[self._index]
        self.stack.append(img)
        return np.array(self.stack)
