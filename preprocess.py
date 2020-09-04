import numpy as np


def down_scale(rgb, crop=True, rescale=True):
    h, w, c = rgb.shape
    if crop:
        rgb = rgb[:, (w-h) // 2:-(w-h) // 2]
        w = h
    if rescale:
        rgb = rgb[::5, ::5]
        w /= 5
        h /= 5
    return rgb


def pre_process(rgb, norm=True):
    h, w, c = rgb.shape
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray #np.expand_dims(gray, axis=2) #.reshape(1, batch, int(h), int(w))


# from torchvision import transforms
#
#
# pre_process = transforms.Compose(
#     [transforms.Grayscale(num_output_channels=1),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.5], std=[0.5])
#      ])