import cv2
import numpy as np


# local imports
from preprocess import pre_process


class Env(object):
    def __init__(self):
        self._index = 0
        self.capacity = int(1e2)
        self._img = []
        self._labs = []
        self.n_imgs = 4
        self.train = False
        self.p_split = .9
        self.batch_size = 32

    def load_video(self, video_path, data_path):
        self._labs = np.loadtxt(data_path)
        self.norm_const = 10 #np.linalg.norm(self._labs)
        self._labs = self._labs / self.norm_const

        vid = cv2.VideoCapture(video_path)
        i = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret or i >= self.capacity: break
            frame = pre_process(frame)
            #cv2.imshow("image", frame)
            #cv2.waitKey(1)
            self._img.append(frame)
            i += 1

        self._img = np.array(self._img)
        self.capacity = self._img.shape[0]
        self._labs = self._labs[self.n_imgs:self.capacity + self.n_imgs]    # neglect the first n_imgs
        _, h, w = self._img.shape
        self.img_stack = np.zeros((self.batch_size, self.n_imgs, h, w))
        self.lab_stack = np.zeros((self.batch_size, 1))

    def prep_eval(self):
        self.train = False

    def prep_train(self):
        self.train = True

    def get_data(self):
        if self.train:
            self._index = np.random.random_integers(low=0,
                                             high=round(self.capacity * self.p_split) - self.n_imgs,
                                             size=self.batch_size)
        else:
            self._index = np.random.random_integers(low=round(self.capacity * self.p_split),
                                             high=self.capacity - self.n_imgs,
                                             size=self.batch_size)

        for i, idx in enumerate(self._index):
            self.img_stack[i][0:self.n_imgs] = self._img[idx:idx+self.n_imgs]
            self.lab_stack[i] = self._labs[idx]

        return self.img_stack, self.lab_stack
