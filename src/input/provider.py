import numpy as np
from input.util import Util
from src.input.distance_util import DistanceUtil
from src.input.filter_provider import FilterProvider
import matplotlib.pyplot as plt
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
        #print(C, X, H)
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

    @staticmethod
    def histogram(image):
        aux = Util.linear_transform(image).astype('B')
        h = np.zeros(256)
        for p in aux.flatten():
            h[p] = h[p] + 1
        print(h)
        return h

    @staticmethod
    def equalize_histogram(image):
        my_h = np.histogram(image, bins=range(257), density=True)
        my_h_acu = my_h[0].copy()
        for i in range(256):
            if i != 0:
                my_h_acu[i] = my_h_acu[i - 1] + my_h[0][i]
        min_h = np.min(my_h_acu)
        for i in range(256):
            if my_h_acu[i] == 1:
                my_h_acu[i] = 255
            else:
                my_h_acu[i] = np.round((my_h_acu[i] - min_h) * 255 / (1 - min_h))
        ans = Util.apply_to_matrix(image, lambda p: my_h_acu[p])
        return ans


# myimg = Util.load_raw('LENA.RAW', (256, 256))
# # Util.save(myimg, 'original')
# # myimg = Util.add_additive_noise_exponential(myimg, scale=100, prob=0.9)
# # myimg = FilterProvider.gauss_blur(myimg, (7, 7), 0.1).astype('B')
# # myimg = FilterProvider.pasa_altos(myimg).astype('B')
# # myimg = Util.to_binary(myimg, 30).astype('B')
# # np.savetxt('../../resources/blur.raw', myimg[:, :, 0])
#
# # Util.save(myimg, 'exp')
# # vec = np.random.exponential(2, 1000)
# # vec = np.random.normal(0, 3, 1000)
# # hist = np.histogram(vec, bins='auto')
# # plt.show()
# # print(hist)
#
# plt.hist(myimg.flatten(), bins=range(256))
# plt.show()
# # vec = np.random.binomial(1, 0.5, (5, 5))
# # print(vec)
# # Provider.save_raw(myimg)
