import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from src.input.distance_util import DistanceUtil

# For each image store (HEIGHT, WIDTH)
KNOWN_SIZES = {'GIRL.RAW': (164, 389), 'BARCO.RAW': (207, 290), 'LENA.RAW': (256, 256), 'LENAX.RAW': (256, 256), 'GIRL2.RAW': (256, 256), 'FRACTAL.RAW': (200, 200)}


class Util:
    @staticmethod
    def load_raw(path):
        name = path.split('/')
        name = name[len(name) - 1]
        print(name)
        size = KNOWN_SIZES[name]
        image = np.fromfile(path, dtype='B').reshape(size[0], size[1])
        aux = np.zeros((image.shape[0], image.shape[1], 3))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                aux[i][j] = [image[i][j], image[i][j], image[i][j]]
        image = aux
        return image.astype('B')

    # returns [image, isColor(boolean)]
    @staticmethod
    def load_image(path):
        image = cv2.imread(path)
        if image is None:
            return Util.load_raw(path)
        (width, height) = (image.shape[0], image.shape[1])

        # Change bgr to rgb color format
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        is_color = False
        for x in range(width):
            for y in range(height):
                is_color = img[x, y, 0] is not img[x, y, 1] or img[x, y, 0] is not img[x, y, 2]
            if is_color:
                break
        if is_color:
            #     return img.astype('float'), True
            # return img.astype('float'), False
            return img.astype('float')
        return img.astype('float')

    # deprecated
    @staticmethod
    def load_gray_scale(path):
        return cv2.imread(path)[:, :, 1]

    # deprecated
    @staticmethod
    def load_color(path):
        return cv2.imread(path)[:, :, :]

    @staticmethod
    def trim(image, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return image[min(x1, x2):max(x1, x2), min(y1, y2):max(y1, y2)]

    @staticmethod
    def average(array):
        count = len(array)
        return float(array.sum()) / max(count, 1)

    @staticmethod
    def get_info(image, p1, p2):
        aux = Util.trim(image, p1, p2)
        count = len(aux)
        return Util.average(aux), count

    @staticmethod
    def save(image, name):
        image = Util.linear_transform(image)
        image = image.astype('short')
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name + ".pbm", img, (cv2.IMWRITE_PXM_BINARY, 0))

    @staticmethod
    def save_raw(image, name='../../resources/blur.raw'):
        image[:, :, 0].astype('B').tofile(name)

    @staticmethod
    def to_binary(image, threshold):
        vfunc = np.vectorize(lambda p: 255 if p > threshold else 0)
        return vfunc(image)

    @staticmethod
    def scalar_prod(image, scalar):
        vfunc = np.vectorize(lambda p: p * scalar)
        return vfunc(image)

    @staticmethod
    def gray_difference(img1, img2):
        (width, height) = img1.shape
        ans = np.zeros(img1.shape)
        for x in range(width):
            for y in range(height):
                ans[x, y] = img1[x, y] - img2[x, y]
        return ans

    # beta
    @staticmethod
    def difference(img1, img2):
        return np.subtract(img1, img2)

    @staticmethod
    def gray_sum(img1, img2):
        (width, height) = img1.shape
        ans = np.zeros(img1.shape)
        for x in range(width):
            for y in range(height):
                ans[x, y] = img1[x, y] + img2[x, y]
        return ans

    @staticmethod
    def sum(img1, img2):
        return Util.element_wise_operation(img1, img2, lambda x, y: x + y)
        # return np.sum(img1, img2)

    @staticmethod
    def multiply(img1, img2):
        return Util.element_wise_operation(img1, img2, lambda x, y: x * y)
        # return np.multiply(img1, img2)

    @staticmethod
    def multiply_not_zero(img1, img2):
        return Util.element_wise_operation(img1, img2, lambda x, y: x if y==0 else x * y)

    @staticmethod
    def linear_transform(image, final_range=(0, 255), to_char=True):
        (final_min, final_max) = final_range
        final_difference = final_max - final_min
        size = image.shape
        width, height = size[0], size[1]
        ans = np.zeros(size)
        min_val = np.min(image)
        max_val = np.max(image)
        if min_val == max_val:
            print('[WARNING] In linear_transform: min_val equals max_val')
            return image
        for x in range(width):
            for y in range(height):
                for z in range(3):
                    ans[x][y][z] = (image[x][y][z] - min_val) * final_difference / (max_val - min_val) + final_min
                    if (to_char):
                        ans[x][y][z] = int(ans[x][y][z])
        if (to_char):
            return ans.astype('B')
        return ans

    @staticmethod
    def gray_hist(image):
        ans = np.zeros(256)
        for p in image.flat:
            ans[p] += 1
        return ans

    @staticmethod
    def negative(image):
        ans = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    ans[i][j][k] = 255 - image[i][j][k]
        return ans

    # ONLY FOR 2D MATRIX
    @staticmethod
    def dynamic_range_compression(image):
        R = image.max()
        c = 255 / np.math.log(1 + R)

        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         ans[i][j] = c * np.math.log(1 + image[i][j])
        # return ans
        def f(val):
            return c * np.math.log(1 + val)

        return Util.apply_to_matrix(image, f, True)

    # ONLY FOR 2D MATRIX
    @staticmethod
    def contrast_increase(image, s1, s2):
        sigma = np.std(image.ravel())
        mean = image.mean()
        r1 = mean - sigma
        r2 = mean + sigma
        if r1 <= 0 or r2 >= 255:
            return image
        m1 = (s1 / r1)
        b1 = 0
        m2 = ((s2 - s1) / (r2 - r1))
        b2 = s1 - m2 * r1
        m3 = (255 - s2) / (255 - r2)
        b3 = s2 - m3 * r2
        # Don't delete this comment, it gives info. about the image
        print('Contrast\nMean: {}, Sigma: {}\nr1: {}, r2: {}\nm1: {}, m2: {}, m3: {}'
              .format(mean, sigma, r1, r2, m1, m2, m3))
        def f(val):
            if 0 <= val <= r1:
                return m1 * val + b1
            elif val <= r2:
                return m2 * val + b2
            else:
                return m3 * val + b3

        return Util.apply_to_matrix(image, f, True)

    @staticmethod
    def standard_deviation(matrix):
        n = matrix.shape[0] * matrix.shape[1]
        mean = matrix.mean()
        sigma = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                sigma += (matrix[i][j] - mean) ** 2
        return np.sqrt(sigma / n)

    @staticmethod
    def gamma_power(image, gamma):
        c = np.power(255, 1 - gamma)
        def f(val):
            return c * np.power(val, gamma)

        return Util.apply_to_matrix(image, f, True)

        # @staticmethod
        # def gaussian_distr(x1, x2):
        # y1 = np.sqrt(-2 * log(x1)) * cos(2 * np.PI * x2)
        # y2 = np.sqrt(-2 * log(x1)) * sin(2 * np.PI * x2)
        # return y1, y2

    @staticmethod
    def box_muller(y1, y2):
        x2 = np.arctan(y2 / y1) / (2 * np.PI)
        # x1 = exp(-(y1 ** 2 + y2 ** 2) / 2)

    # (my_image, is_color) = Util.load_image('../../resources/lena.ascii.pbm')
    # print(Util.gray_hist(my_image[0]))
    @staticmethod
    def binary_matrix(shape, prob=0.5):
        return np.random.binomial(1, prob, shape)

    @staticmethod
    def add_noise_exponential(image, scale=1, prob=1):
        aux = Util.multiply(
            np.random.exponential(scale, image.shape),
            Util.binary_matrix(image.shape, prob)
        )
        aux = Util.multiply_not_zero(image, aux)
        return aux

    @staticmethod
    def add_noise_rayleigh(image, scale=1, prob=1):
        # return Util.multiply(image, np.random.rayleigh(scale=scale, size=image.shape))
        aux = Util.multiply(
            np.random.rayleigh(scale, image.shape),
            Util.binary_matrix(image.shape, prob)
        )
        aux = Util.multiply_not_zero(image, aux)
        return aux

    @staticmethod
    def add_additive_noise_normal(image, mu=0, sigma=20, prob=1):
        return Util.sum(image, Util.multiply(
            np.random.normal(mu, sigma, image.shape),
            Util.binary_matrix(image.shape, prob)
        ))

    @staticmethod
    def single_comino_and_sugar(value, p0, p1):
        r = np.random.random()
        if r <= p0:
            return 0
        elif r >= p1:
            return 255
        return value

    @staticmethod
    def add_comino_and_sugar_noise(image, p0=0.1, p1=0.9):
        # return Util.apply_to_matrix(image, lambda img: Util.single_comino_and_sugar(img, p0, p1))
        ans = np.copy(image).astype(float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ans[i][j][0] = Util.single_comino_and_sugar(image[i][j][0], p0, p1)
                if ans[i][j][0] == 0 or ans[i][j][0] == 255:
                    ans[i][j][1] = ans[i][j][0]
                    ans[i][j][2] = ans[i][j][0]
        return ans

    @staticmethod
    def apply_to_matrix(matrix, func, independent_layer=False, two_dim=False):
        negative = np.copy(matrix).astype(float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if two_dim:
                    negative[i][j] = func(negative[i][j])
                elif independent_layer:
                    for k in range(matrix.shape[2]):
                        negative[i][j][k] = func(negative[i][j][k])
                else:
                    aux = func(negative[i][j][0])
                    for k in range(matrix.shape[2]):
                        negative[i][j][k] = aux
        return negative

    @staticmethod
    def apply_to_matrix_with_position(matrix, func, independent_layer=False, two_dim=False):
        negative = np.copy(matrix).astype(float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if two_dim:
                    negative[i][j] = func(negative[i][j], i, j)
                elif independent_layer:
                    for k in range(matrix.shape[2]):
                        negative[i][j][k] = func(negative[i][j][k], i, j)
                else:
                    aux = func(negative[i][j][0], i, j)
                    for k in range(matrix.shape[2]):
                        negative[i][j][k] = aux
        return negative

    @staticmethod
    def element_wise_operation(matrix1, matrix2, func, independent_layer=False):
        ans = np.zeros(matrix1.shape).astype(float)
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):
                if independent_layer:
                    for k in range(matrix1.shape[2]):
                        ans[i][j][k] = func(matrix1[i][j][k], matrix2[i][j][k])
                else:
                    aux = func(matrix1[i][j][0], matrix2[i][j][0])
                    for k in range(matrix1.shape[2]):
                        ans[i][j][k] = aux
        return ans

# img = Util.load_raw('LENA.RAW')
# img = Util.apply_to_matrix(img, lambda x: [x,x,x], two_dim=True)
# Util.save(img, 'leemos_raw')
