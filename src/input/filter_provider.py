import numpy as np

from src.input.util import Util


class FilterProvider:
    @staticmethod
    def blur(image, size):
        mask = np.zeros(size)
        mask = Util.apply_to_matrix(mask, lambda val: 1 / mask.flatten().size, two_dim=True)
        print(mask)
        return Util.sliding_window(image, mask)
