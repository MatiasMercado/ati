import cv2
import numpy as np

from src.input.distance_util import DistanceUtil
import matplotlib.pyplot as plt


class Util:
    @staticmethod
    def load_raw(path, size):
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
            return img.astype('float'), True
        return img.astype('float'), False

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
        print(np.min(image), np.max(image))
        image = Util.linear_transform(image)
        print(np.min(image), np.max(image))
        image = image.astype('short')
        print(np.min(image), np.max(image))
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name + ".pbm", img, (cv2.IMWRITE_PXM_BINARY, 0))

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
    def linear_transform(image, final_range=(0, 255)):
        (final_min, final_max) = final_range
        final_difference = final_max - final_min
        size = image.shape
        width, height = size[0], size[1]
        ans = np.zeros(size)
        min_val = np.min(image)
        max_val = np.max(image)
        for x in range(width):
            for y in range(height):
                for z in range(3):
                    ans[x][y][z] = (image[x][y][z] - min_val) * final_difference / (max_val - min_val) + final_min
                    if (ans[x][y][z] >= 255):
                        print(image[x][y][z], ans[x][y][z])
        return ans

    @staticmethod
    def gray_hist(image):
        ans = np.zeros(256)
        for p in image.flat:
            ans[p] += 1
        return ans

    @staticmethod
    def negative(image):
        # vfunc = np.vectorize(lambda p: 255 - p)
        # return vfunc(image)
        negative = np.copy(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    negative[i][j][k] = 255 - negative[i][j][k]
        return negative

        # @staticmethod
        # def gaussian_distr(x1, x2):
        # y1 = np.sqrt(-2 * log(x1)) * cos(2 * np.PI * x2)
        # y2 = np.sqrt(-2 * log(x1)) * sin(2 * np.PI * x2)
        # return y1, y2

    @staticmethod
    def box_muller(y1, y2):
        x2 = np.arctan(y2 / y1) / (2 * np.PI)
        # x1 = exp(-(y1 ** 2 + y2 ** 2) / 2)

    @staticmethod
    def sliding_window(image, mask, border_policy=0):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                for z in range(image.shape[2]):
                    ans[x, y, z] = Util.apply_mask(image[:, :, z], (x, y), mask)
        return ans

    @staticmethod
    def apply_mask(image, center, mask, border_policy=0):
        (image_width, image_height) = image.shape
        (center_x, center_y) = center
        (mask_width, mask_height) = mask.shape
        acu = 0
        for x in range(mask_width):
            image_x = center_x - int(mask_width / 2) + x
            if image_x >= image_width:
                image_x -= mask_width
            elif image_x < 0:
                image_x += mask_width
            for y in range(mask_height):
                image_y = center_y - int(mask_height / 2) + y
                if image_y >= image_height:
                    image_y -= mask_height
                elif image_y < 0:
                    image_y += mask_height
                acu += mask[x][y] * image[image_x][image_y]
        return acu

    # (my_image, is_color) = Util.load_image('../../resources/lena.ascii.pbm')
    # print(Util.gray_hist(my_image[0]))
    @staticmethod
    def binary_matrix(shape, prob=0.5):
        return np.random.binomial(1, prob, shape)

    @staticmethod
    def add_additive_noise_exponential(image, scale=1, prob=0.5):
        aux = Util.multiply(
            np.random.exponential(scale, image.shape),
            Util.binary_matrix(image.shape, prob)
        )
        return Util.sum(image, aux)

    @staticmethod
    def add_additive_noise_normal(image, mu=0, sigma=1, prob=0.5):
        return Util.sum(image, Util.multiply(
            np.random.normal(mu, sigma, image.shape),
            Util.binary_matrix(image.shape, prob)
        ))

    @staticmethod
    def single_comino_and_sugar(value, prob):
        r = np.random.random()
        if r > prob:
            return value
        if r > prob / 2:
            return 255
        return 0

    @staticmethod
    def add_comino_and_sugar_noise(image, prob=0.5):
        ret = Util.apply_to_matrix(image, lambda p: Util.single_comino_and_sugar(p, prob))
        return ret

    @staticmethod
    def apply_to_matrix(matrix, func, independent_layer=False, two_dim=False):
        negative = np.copy(matrix)
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
    def apply_to_matrix_with_position(matrix, func, independent_layer=False):
        negative = np.copy(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if independent_layer:
                    for k in range(matrix.shape[2]):
                        negative[i][j][k] = func(negative[i][j][k], i, j)
                else:
                    aux = func(negative[i][j][0], i, j)
                    for k in range(matrix.shape[2]):
                        negative[i][j][k] = aux
        return negative

    @staticmethod
    def element_wise_operation(matrix1, matrix2, func, independent_layer=False):
        negative = np.copy(matrix1)
        print('start')
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):
                if independent_layer:
                    for k in range(matrix1.shape[2]):
                        negative[i][j][k] = func(matrix1[i][j][k], matrix2[i][j][k])
                else:
                    aux = func(matrix1[i][j][0], matrix2[i][j][0])
                    for k in range(matrix1.shape[2]):
                        negative[i][j][k] = aux
        print('max', np.max(negative))
        return negative

    @staticmethod
    def add_additive_noise_normal(image, mu=0, sigma=1):
        Util.sum(image, np.random.rayleigh(mu, sigma, image.shape))

# img = Util.load_raw('LENA.RAW')
