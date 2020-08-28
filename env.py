import cv2
import numpy as np


# local imports
from preprocess import pre_process


class Env(object):
    def __init__(self):
        self.capacity = int(1e4)
        self._img = []
        self._labs = []
        self.n_imgs = 4
        self.train = False
        self.split_perc = .9
        self.batch_size = 32
        self.norm_const = 10

    def load_labels(self, data_path):
        self._labs = np.loadtxt(data_path)
        #self.norm_const np.linalg.norm(self._labs)
        self._labs = self._labs / self.norm_const

    def load_video(self, video_path):
        vid = cv2.VideoCapture(video_path)
        i = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret or i >= self.capacity: break
            frame = pre_process(frame)
            self._img.append(frame)
            i += 1

        self._img = np.array(self._img)
        self.capacity = self._img.shape[0]
        self._labs = self._labs[self.n_imgs:self.capacity + self.n_imgs]    # neglect the first n_imgs
        _, h, w = self._img.shape
        self.img_stack = np.zeros((self.batch_size, self.n_imgs, h, w))
        self.lab_stack = np.zeros((self.batch_size, 1))
        self._indeces = np.arange(self.capacity - self.n_imgs)

    def shuffle_data(self):
        np.random.shuffle(self._indeces)

    def prep_eval(self):
        self.train = False

    def prep_train(self):
        self.train = True

    def get_data(self, show=False):
        p_split = int(np.floor(self.capacity * self.split_perc) - self.n_imgs)

        if self.train:
            index_choice = np.random.choice(self._indeces[:p_split], self.batch_size)
        else:
            index_choice = np.random.choice(self._indeces[p_split:], self.batch_size)

        for i, idx in enumerate(index_choice):
            self.img_stack[i][0:self.n_imgs] = self._img[idx:idx+self.n_imgs]
            self.lab_stack[i] = self._labs[idx]
            if show:
                for j in range(self.n_imgs):
                    cv2.imshow("image", np.expand_dims(self._img[idx+j], 2))
                    cv2.waitKey(10)

        return self.img_stack, self.lab_stack

    def get_img(self, i):
        return self._img[i:i+self.n_imgs]
