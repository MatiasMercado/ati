import cv2
import numpy as np

from src.input.distance_util import DistanceUtil


class Util:
    @staticmethod
    def load_gray_scale(path):
        return cv2.imread(path)[:, :, 1]

    @staticmethod
    def trim(image, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return image[min(x1, x2):max(x1, x2), min(y1, y2):max(y1, y2)]

    @staticmethod
    def get_info(image, p1, p2):
        aux = Util.trim(image, p1, p2)
        count = len(aux)
        return float(aux.sum()) / max(count, 1), count

    @staticmethod
    def save(image, name):
        cv2.imwrite(name + ".pbm", image, (cv2.IMWRITE_PXM_BINARY, 0))



PATH = '../../resources/lena.ascii.pbm'
img = Util.load_gray_scale(PATH)
print(img.shape)
img = Util.trim(img, (30, 40), (60, 85))
print(img.shape)
info = Util.get_info(img, (3, 3), (7, 7))
print(info)
circle = np.zeros((200, 200), dtype=np.short)
square = np.zeros((200, 200), dtype=np.short)
center = (100, 100)
for x in range(200):
    for y in range(200):
        if DistanceUtil.euclidean_distance_lower_than(center, (x, y), 40):
            circle[x, y] = 255

for x in range(200):
    for y in range(200):
        if DistanceUtil.chebyshev_distance_lower_than(center, (x, y), 40):
            square[x, y] = 255

Util.save(circle, "circle")
Util.save(square, "square")
