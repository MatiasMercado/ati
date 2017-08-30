import cv2
import numpy as np

from src.input.distance_util import DistanceUtil


class Util:
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
        # vfunc = np.vectorize(lambda p: 255 - p)
        # return vfunc(image)
        negative = np.copy(image)
        for i in range(512):
            for j in range(512):
                negative[i][j] = 255 - negative[i][j]
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
        (image_width, image_height) = image.shape
        for x in range(image_width):
            for y in range(image_height):
                ans[x, y] = Util.apply_mask(image, (x, y), mask)
        return ans

    @staticmethod
    def apply_mask(image, center, mask, border_policy=0):
        (image_width, image_height) = image.shape
        (center_x, center_y) = center
        (mask_width, mask_height) = mask.shape
        acu = 0
        for x in mask_width:
            image_x = center_x - int(mask_width / 2) + x
            if image_x >= image_width:
                image_x -= mask_width
            elif image_x < 0:
                image_x += mask_width
            for y in mask_height:
                image_y = center_y - int(mask_height / 2) + y
                if image_y >= image_height:
                    image_y -= mask_height
                elif image_y < 0:
                    image_y += mask_height
                acu += mask(x, y) * image(image_x, image_y)
        return acu

# (my_image, is_color) = Util.load_image('../../resources/lena.ascii.pbm')
# print(Util.gray_hist(my_image[0]))
