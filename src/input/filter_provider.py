import numpy as np

from src.input.util import Util


class FilterProvider:
    @staticmethod
    def blur(image, size):
        mask = np.zeros(size)
        mask = Util.apply_to_matrix(mask, lambda val: 1 / mask.flatten().size, two_dim=True)
        print(mask)
        return Util.sliding_window(image, mask)

    @staticmethod
    def gauss_blur(image, size, sigma):
        mask = np.zeros(size)
        mask = Util.apply_to_matrix_with_position(mask, lambda val, x, y: np.exp(-(x ** 2 + y ** 2) / (sigma ** 2)) / (
            2 * np.pi * sigma ** 2), two_dim=True)
        mask_sum = mask.sum()
        print('gasuss sum:', mask_sum)
        mask = Util.apply_to_matrix(mask, lambda val: val / mask_sum, two_dim=True)
        mask_sum = mask.sum()
        print('gasuss sum:', mask_sum)
        print(mask)
        return Util.sliding_window(image, mask)

    @staticmethod
    def pasa_altos(image):
        mask = np.zeros((3, 3))
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    mask[x][y] = 8/9
                else:
                    mask[x][y] = -1/9
        print(mask)
        return Util.sliding_window(image, mask)
