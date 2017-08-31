import numpy as np

from src.input.util import Util


class FilterProvider:
    @staticmethod
    def blur(image, size):
        mask = np.zeros(size)
        mask = Util.apply_to_matrix(mask, lambda val: 1 / mask.flatten().size, two_dim=True)
        print(mask)
        return FilterProvider.sliding_window(image, mask)

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
        return FilterProvider.sliding_window(image, mask)

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
        return FilterProvider.sliding_window(image, mask)

    @staticmethod
    def sliding_window_median(image, mask, weighted=False, border_policy=0):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                for z in range(image.shape[2]):
                    ans[x, y, z] = FilterProvider.apply_mask_median(image[:, :, z], (x, y), mask, weighted=weighted)
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

    @staticmethod
    def apply_mask_median(image, center, mask, weighted=False, border_policy=0):
        (image_width, image_height) = image.shape
        (center_x, center_y) = center
        (mask_width, mask_height) = mask.shape
        vec = []
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
                if weighted:
                    for i in range(mask[x][y]):
                        vec.append(image[image_x][image_y])
                else:
                    vec.append(image[image_x][image_y])
        return np.median(vec)

    @staticmethod
    def sliding_window(image, mask, border_policy=0):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                for z in range(image.shape[2]):
                    ans[x, y, z] = FilterProvider.apply_mask(image[:, :, z], (x, y), mask)
        return ans

