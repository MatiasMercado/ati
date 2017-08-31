import numpy as np

from src.input.distance_util import DistanceUtil
from src.input.filter_provider import FilterProvider
from src.input.util import Util


class Provider:
    @staticmethod
    def gray_gradient(size, min, max):
        (width, height) = size
        img = np.zeros(size)
        for x in range(width):
            for y in range(height):
                img[x, y] = x * (max - min) / (width - 1) + min
        return img

    @staticmethod
    def color_gradient(size):
        (width, height) = size
        img = np.zeros((width, height, 3), dtype=np.short)

        for x in range(width):
            for y in range(height):
                img[x, y] = Provider.hsv_to_rgb(x * 360 / (width - 1), 1, y / (height - 1))

        # pixel = [0, 0, 0]
        # for c in range(3):
        #     for x in range(width):
        #         for y in range(height):
        #             img[x + c * width, y] = pixel.copy()
        #         pixel[c] += 1
        #         print(pixel)
        return img

    @staticmethod
    def get_rgb_p(C, X, H):
        print(C, X, H)
        h = int(H / 60)
        if h == 0:
            return C, X, 0
        if h == 1:
            return X, C, 0
        if h == 2:
            return 0, C, X
        if h == 3:
            return 0, X, C
        if h == 4:
            return X, 0, C
        return C, 0, X

    @staticmethod
    def hsv_to_rgb(H, S, V):
        C = V * S
        X = C * (1 - abs(((H / 60) % 2) - 1))
        m = V - C

        (Rp, Gp, Bp) = Provider.get_rgb_p(C, X, H)

        return [(Rp + m) * 255, (Gp + m) * 255, (Bp + m) * 255]

    @staticmethod
    def draw_circle(size, radius):
        (width, height) = size
        circle = np.zeros((width, height, 3), dtype=np.short)
        center = (width / 2, height / 2)
        for x in range(width):
            for y in range(height):
                for z in range(3):
                    if DistanceUtil.euclidean_distance_lower_than(center, (x, y), radius):
                        circle[x, y, z] = 255
        return circle

    @staticmethod
    def draw_square(size, side):
        (width, height) = size
        square = np.zeros(size, dtype=np.short)
        center = (width / 2, height / 2)
        for x in range(width):
            for y in range(height):
                if DistanceUtil.chebyshev_distance_lower_than(center, (x, y), side):
                    square[x, y] = 255
        return square


myimg = Util.load_raw('LENA.RAW', (256, 256))
print(myimg.shape)
# Util.save(myimg, 'original')
myimg = Util.add_additive_noise_exponential(myimg, scale=5, prob=0.7)
myimg = FilterProvider.blur(myimg, (10, 10))
np.savetxt('blur', myimg[:, :, 0])


# print(np.max(myimg))
# print(np.min(myimg))
# Util.save(myimg, 'exp')
# vec = np.random.exponential(2, 1000)
# vec = np.random.normal(0, 3, 1000)
# hist = np.histogram(vec, bins='auto')
# # plt.show()
# # print(hist)
# vec = np.random.binomial(1, 0.5, (5, 5))
# # print(vec)
