import numpy as np

from src.input.util import Util

WEIGHTED_MEDIAN_MASK = np.matrix([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])
class FilterProvider:
    @staticmethod
    def blur(image, size):
        mask = np.zeros(size)
        mask = Util.apply_to_matrix(mask, lambda val: 1 / mask.flatten().size, two_dim=True)
        return FilterProvider.__sliding_window(image, mask)

    @staticmethod
    def gauss_blur(image, size, sigma):
        mask = np.zeros(size)

        def gauss_function(x, y):
            return np.exp(-(x ** 2 + y ** 2) / (sigma ** 2)) / (2 * np.pi * sigma ** 2)
        # Create Mask
        for i in range(size[0]):
            for j in range(size[1]):
                        mask_index_i = i - int(mask.shape[0]/2) # Go from -rows/2 to +rows/2
                        mask_index_j = j - int(mask.shape[1]/2) # Go from -cols/2 to +cols/2
                        mask[i][j] = gauss_function(mask_index_i, mask_index_j)

        return FilterProvider.__sliding_window(image, mask)

    @staticmethod
    def pasa_altos(image):
        mask = np.zeros((3, 3))
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    mask[x][y] = 8/9
                else:
                    mask[x][y] = -1/9
        return FilterProvider.__sliding_window(image, mask)

    @staticmethod
    def median_filter(image, mask=WEIGHTED_MEDIAN_MASK, weighted=False, border_policy=0):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                for z in range(image.shape[2]):
                    ans[x, y, z] = FilterProvider.__apply_mask_median(image[:, :, z], (x, y), mask, weighted=weighted)
        return ans

    @staticmethod
    def __apply_mask(image, center, mask, border_policy=0):
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
                acu += mask[x, y] * image[image_x][image_y]
        return acu

    @staticmethod
    def __apply_mask_median(image, center, mask, weighted=False, border_policy=0):
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
                    for i in range(mask[x,y]):
                        vec.append(image[image_x][image_y])
                else:
                    vec.append(image[image_x][image_y])
        return np.median(vec)

    @staticmethod
    def __sliding_window(image, mask, border_policy=0):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                for z in range(image.shape[2]):
                    ans[x, y, z] = FilterProvider.__apply_mask(image[:, :, z], (x, y), mask)
        return ans

    @staticmethod
    def y_border(image, weighted=False):
        mask = np.zeros((3, 3))
        for x in range(3):
            for y in range(3):
                if x == 0:
                    if y == 1 and weighted:
                        mask[x][y] = 2
                    else:
                        mask[x][y] = 1
                elif x == 2:
                    if y == 1 and weighted:
                        mask[x][y] = -2
                    else:
                        mask[x][y] = -1
                else:
                    mask[x][y] = 0
        return FilterProvider.__sliding_window(image, mask)

    @staticmethod
    def x_border(image, weighted=False):
        mask = np.zeros((3, 3))
        for x in range(3):
            for y in range(3):
                if y == 0:
                    if x == 1 and weighted:
                        mask[x][y] = 2
                    else:
                        mask[x][y] = 1
                elif y == 2:
                    if x == 1 and weighted:
                        mask[x][y] = -2
                    else:
                        mask[x][y] = -1
                else:
                    mask[x][y] = 0
        return FilterProvider.__sliding_window(image, mask)


# img = Util.load_raw('LENA.RAW')
# # img = Util.apply_to_matrix(img, lambda x: [x,x,x], two_dim=True)
# img_x = FilterProvider.x_border(img, False)
# img_x = Util.apply_to_matrix(img_x, lambda p: np.abs(p))
# Util.save_raw(img_x, 'x_lena')
#
# img_y = FilterProvider.y_border(img, False)
# img_y = Util.apply_to_matrix(img_y, lambda p: np.abs(p))
# Util.save_raw(img_y, 'y_lena')
