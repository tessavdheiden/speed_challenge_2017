import cv2
import numpy as np


class Env(object):
    def __init__(self):
        self._index = 0
        self.capacity = 1e2
        self._img = []
        self.n_imgs = 4
        self.stack = [None] * self.n_imgs

    def load_video(self, video_path, data_path):
        vid = cv2.VideoCapture(video_path)
        i = 0
        while vid.isOpened():
            ret, frame = vid.read()
            i += 1
            if ret and i < self.capacity:
                cv2.imshow("image", frame)
                cv2.waitKey(1)
                self._img.append(frame)
            else:
                break

        self.stack = [self._img[0]] * self.n_imgs

    def get_data(self):
        self.stack.pop(0)
        self._index += 1
        if self._index >= self.capacity - self.n_imgs:
            self._index = 0

        img = self._img[self._index]
        self.stack.append(img)
        return np.array(self.stack), 0
