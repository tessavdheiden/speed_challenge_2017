import torch
import torch.nn as nn
import numpy as np

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, n_imgs):
        super(MLPNetwork, self).__init__()
        self.cnn = ConvolutionalUnit(n_imgs=n_imgs).to(torch.double)

    def forward(self, X):
        return self.cnn(X)


class ConvolutionalUnit(nn.Module):
    def __init__(self, n_imgs):
        super(ConvolutionalUnit, self).__init__()
        self.hidden_dim = 64
        self.out_dim = 1
        # layer_helper = lambda i, k, s: (i - k) / s + 1
        ch_in = int(np.log2(n_imgs))
        ch = [2**i for i in range(ch_in, ch_in+7)]
        self.ch_out = ch[-1]
        self.cnn = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(ch[1], ch[2], kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(ch[4], ch[5], kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(ch[5], ch[6], kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.fc = nn.Sequential(nn.Linear(self.ch_out, 100), nn.ReLU(), nn.Linear(100, 1))
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        batch, channels, w, h = x.shape
        x = self.cnn(x.view(batch, channels, w, h))
        x = x.view(-1, self.ch_out)
        return self.fc(x)