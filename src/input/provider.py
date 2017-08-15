import numpy as np

from src.input.util import Util


class Provider:
    @staticmethod
    def gray_gradient(size, min, max):
        (width, height) = size
        img = np.zeros((width, height), dtype=np.short)
        for x in range(width):
            for y in range(height):
                img[x, y] = x * (max - min) / (width - 1) + min
        return img

aux = Provider.gray_gradient((255, 255), 0, 255)
Util.save(aux, "gradient")
