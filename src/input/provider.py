import numpy as np

from src.input.util import Util
from src.input.vector_util import VectorUtil


class Provider:
    @staticmethod
    def gray_gradient(size=(256, 256), min=0, max=255):
        (width, height) = size
        img = np.zeros(size)
        for x in range(width):
            for y in range(height):
                img[x, y] = x * (max - min) / (width - 1) + min
        return img

    @staticmethod
    def color_gradient(size=(256, 256)):
        (width, height) = size
        img = np.zeros((width, height, 3), dtype=np.short)

        for x in range(width):
            for y in range(height):
                img[x, y] = Provider.__hsv_to_rgb(x * 360 / (width - 1), 1, y / (height - 1))

        # pixel = [0, 0, 0]
        # for c in range(3):
        #     for x in range(width):
        #         for y in range(height):
        #             img[x + c * width, y] = pixel.copy()
        #         pixel[c] += 1
        #         print(pixel)
        return img

    @staticmethod
    def __get_rgb_p(C, X, H):
        # print(C, X, H)
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
    def __hsv_to_rgb(H, S, V):
        C = V * S
        X = C * (1 - abs(((H / 60) % 2) - 1))
        m = V - C

        (Rp, Gp, Bp) = Provider.__get_rgb_p(C, X, H)

        return [(Rp + m) * 255, (Gp + m) * 255, (Bp + m) * 255]

    @staticmethod
    def draw_circle(size=(256, 256), radius=60):
        (width, height) = size
        circle = np.zeros((width, height, 3), dtype=np.short)
        center = (width / 2, height / 2)
        for x in range(width):
            for y in range(height):
                for z in range(3):
                    if VectorUtil.euclidean_distance_lower_than(center, (x, y), radius):
                        circle[x, y, z] = 255
        return circle

    @staticmethod
    def draw_diana(size=(256, 256), radiusList=[(40, 255), (50, 100), (70, 150), (80, 200)]):
        (height, width) = size
        circle = np.zeros((height, width, 3), dtype=np.short)
        center = (height / 2, width / 2)
        for x in range(width):
            for y in range(height):
                for z in range(3):
                    for radius in radiusList:
                        if VectorUtil.euclidean_distance_lower_than(center, (x, y), radius[0]):
                            circle[x, y, z] = radius[1]
                            break
        return circle


    @staticmethod
    def get_circle_coordinates(radius, center, step=0.05):
        (x_c, y_c) = center
        ret = []
        for x in np.arange(-radius, radius, step):
            new_x = x_c + int(x)
            new_y = y_c + int(np.sqrt(radius * radius - x * x))
            if (new_x, new_y) not in ret:
                ret.append((new_x, new_y))
        for x in np.arange(radius - step, -radius + step, -step):
            new_x = x_c + int(x)
            new_y = y_c - int(np.sqrt(radius * radius - x * x))
            if (new_x, new_y) not in ret:
                ret.append((new_x, new_y))
        return ret

    @staticmethod
    def draw_square(size=(256, 256), side=50):
        (width, height) = size
        square = np.zeros((width, height, 3), dtype=np.short)
        center = (width / 2, height / 2)
        for x in range(width):
            for y in range(height):
                for z in range(3):
                    if VectorUtil.chebyshev_distance_lower_than(center, (x, y), side):
                        square[x, y, z] = 255
        return square

    @staticmethod
    def histogram(image):
        aux = Util.linear_transform(image).astype('B')
        h = np.zeros(256)
        for p in aux.flatten():
            h[p] = h[p] + 1
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
        ans = Util.apply_to_matrix(image, lambda p: my_h_acu[p.astype(int)])
        return ans
