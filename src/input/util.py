import cv2
import numpy as np

from src.input.distance_util import DistanceUtil


class Util:
    # returns [iamge, isColor(boolean)]
    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        (width, height) = img.shape
        is_color = False
        for x in range(width):
            for y in range(height):
                is_color = img[x, y, 0] != img[x, y, 1] or img[x, y, 0] != img[x, y, 2]
            if is_color:
                break
        if is_color:
            return img, True
        return img[:, :, 1], False

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
        cv2.imwrite(name + ".pbm", image, (cv2.IMWRITE_PXM_BINARY, 0))

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
        return np.sum(img1, img2)

    @staticmethod
    def linear_transform(image, final_range=(0, 255)):
        (final_min, final_max) = final_range
        final_difference = final_max - final_min
        size = image.shape
        (width, height) = size
        ans = np.zeros(size)
        min_val = min(image)
        max_val = max(image)
        for x in range(width):
            for y in range(height):
                ans[x, y] = (image[x, y] - min_val) * final_difference / (max_val - min_val) + final_min
        return ans

    @staticmethod
    def gray_hist(image):
        ans = np.zeros(256)
        for p in image.flat:
            ans[p] += 1
        return ans

    @staticmethod
    def negative(image):
        vfunc = np.vectorize(lambda p: 255 - p)
        return vfunc(image)

    # @staticmethod
    # def gaussian_distr(x1, x2):
        # y1 = np.sqrt(-2 * log(x1)) * cos(2 * np.PI * x2)
        # y2 = np.sqrt(-2 * log(x1)) * sin(2 * np.PI * x2)
        # return y1, y2

    @staticmethod
    def box_muller(y1, y2):
        x2 = np.arctan(y2 / y1) / (2 * np.PI)
        # x1 = exp(-(y1 ** 2 + y2 ** 2) / 2)
