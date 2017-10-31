import numpy as np

from src.input.util import Util

WEIGHTED_MEDIAN_MASK = np.matrix([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])


SUSAN_MASK = np.matrix([[0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0]])


SUSAN_BORDER_DETECTOR = 0
SUSAN_CORNER_DETECTOR = 1
SUSAN_BORDER_CORNER_DETECTOR = 2
SUSAN_MASK_SIZE = 37
SUSAN_BORDER_POINT = 1
SUSAN_CORNER_POINT = 2

LORENTZ_BORDER_DETECTOR = 0
LECLERC_BORDER_DETECTOR = 1
ISOTROPIC_BORDER_DETECTOR = 2


class FilterProvider:
    @staticmethod
    def blur(image, size, independent_layer):
        mask = np.zeros(size)
        mask = Util.apply_to_matrix(mask, lambda val: 1 / mask.flatten().size, two_dim=True)
        return FilterProvider.sliding_window(image, mask, independent_layer)

    @staticmethod
    def gauss_blur(image, size, sigma, independent_layer=False, two_dim=False):
        mask = np.zeros(size)

        def gauss_function(x, y):
            return np.exp(-(x ** 2 + y ** 2) / (sigma ** 2)) / (2 * np.pi * sigma ** 2)

        # Create Mask
        for i in range(size[0]):
            for j in range(size[1]):
                mask_index_i = i - int(mask.shape[0] / 2)  # Go from -rows/2 to +rows/2
                mask_index_j = j - int(mask.shape[1] / 2)  # Go from -cols/2 to +cols/2
                mask[i][j] = gauss_function(mask_index_i, mask_index_j)

        return FilterProvider.sliding_window(image, mask, independent_layer, two_dim=two_dim)

    @staticmethod
    def pasa_altos(image):
        mask = np.zeros((3, 3))
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    mask[x][y] = 8 / 9
                else:
                    mask[x][y] = -1 / 9
        return FilterProvider.sliding_window(image, mask)

    @staticmethod
    def median_filter(image, mask=WEIGHTED_MEDIAN_MASK, weighted=False, independent_layer=False, border_policy=0):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                if independent_layer:
                    for z in range(image.shape[2]):
                        ans[x, y, z] = FilterProvider.__apply_mask_median(image[:, :, z], (x, y), mask,
                                                                          weighted=weighted)
                else:
                    aux = FilterProvider.__apply_mask_median(image[:, :, 0], (x, y), mask, weighted=weighted)
                    for z in range(image.shape[2]):
                        ans[x, y, z] = aux
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
                    for i in range(mask[x, y]):
                        vec.append(image[image_x][image_y])
                else:
                    vec.append(image[image_x][image_y])
        return np.median(vec)

    @staticmethod
    def __apply_mask_susan(image, center, mask, detector_type, delta):
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
                if mask[x, y] == 1 and np.abs(image[image_x][image_y] - image[center_x][center_y]) < 27:
                    acu += 1
        s = 1 - (acu / SUSAN_MASK_SIZE)
        if 0.75 - delta <= s <= 0.75 + delta and \
                (detector_type == SUSAN_CORNER_DETECTOR or
                         detector_type == SUSAN_BORDER_CORNER_DETECTOR):
            return SUSAN_CORNER_POINT
        elif 0.5 - delta <= s <= 0.5 + delta and \
                (detector_type == SUSAN_BORDER_DETECTOR or
                         detector_type == SUSAN_BORDER_CORNER_DETECTOR):
            return SUSAN_BORDER_POINT
        else:
            return 0

    @staticmethod
    def sliding_window(image, mask, independent_layer=False, border_policy=0, two_dim=False, threads=1):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        if two_dim:
            for x in range(image_width):
                for y in range(image_height):
                    ans[x, y] = FilterProvider.__apply_mask(image, (x, y), mask)
        else:
            if threads == 1:
                for x in range(image_width):
                    for y in range(image_height):
                        if independent_layer:
                            for z in range(image.shape[2]):
                                ans[x, y, z] = FilterProvider.__apply_mask(image[:, :, z], (x, y), mask)
                        else:
                            aux = FilterProvider.__apply_mask(image[:, :, 0], (x, y), mask)
                            for z in range(image.shape[2]):
                                ans[x, y, z] = aux

        return ans

    @staticmethod
    def susan_sliding_window(image, independent_layer=False, detector_type=SUSAN_CORNER_DETECTOR,
                             delta=0.05):
        mask = SUSAN_MASK
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]
        for x in range(image_width):
            for y in range(image_height):
                if independent_layer:
                    for z in range(image.shape[2]):
                        ans[x, y, z] = FilterProvider.__apply_mask_susan(image[:, :, z], (x, y), mask, detector_type,
                                                                         delta)
                else:
                    aux = FilterProvider.__apply_mask_susan(image[:, :, 0], (x, y), mask, detector_type, delta)
                    for z in range(image.shape[2]):
                        ans[x, y, z] = aux
        return ans

    @staticmethod
    def border(image, weighted=False, direction=0, independent_layer=False):
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
        mask = FilterProvider.rotate_matrix(mask, direction)
        aux = FilterProvider.sliding_window(image, mask, independent_layer)
        return Util.apply_to_matrix(aux, lambda p: np.abs(p), independent_layer)

    @staticmethod
    def four_directions_border(image, weighted=False, merge_function=lambda p1, p2: p1 if p1 > p2 else p2):
        # copy = image.copy()
        ret = np.zeros(image.shape)
        for i in range(4):
            ret = Util.element_wise_operation(ret, FilterProvider.border(
                image, weighted=weighted, direction=i), merge_function)
        return ret

    @staticmethod
    def directional_border_detector(image, weighted=False, directions=[0, 1, 2, 3],
                                    merge_function=lambda p1, p2: p1 if p1 > p2 else p2, independent_layer=False):
        # copy = image.copy()
        ret = np.zeros(image.shape)
        for i in directions:
            print(i)
            ret = Util.element_wise_operation(ret, FilterProvider.border(
                image, weighted=weighted, direction=i, independent_layer=independent_layer), merge_function, independent_layer)
        return ret

    @staticmethod
    def rotate_matrix(matrix, times=1):
        for i in range(times):
            aux = matrix[0, 0]
            matrix[0, 0] = matrix[1, 0]
            matrix[1, 0] = matrix[2, 0]
            matrix[2, 0] = matrix[2, 1]
            matrix[2, 1] = matrix[2, 2]
            matrix[2, 2] = matrix[1, 2]
            matrix[1, 2] = matrix[0, 2]
            matrix[0, 2] = matrix[0, 1]
            matrix[0, 1] = aux
        return matrix

    @staticmethod
    def anisotropic_filter(image, tmax, m, sigma, independent_layer=False):
        aux = image
        # 0: Lorentz, # 1: Leclerc, 2: Isotropic
        if m == LORENTZ_BORDER_DETECTOR:
            g = FilterProvider.__lorentz
        elif m == LECLERC_BORDER_DETECTOR:
            g = FilterProvider.__leclerc
        else:
            g = FilterProvider.__isotropic

        def anisotropic_single_filter(img, i, j, k, t, g=g):
            height = img.shape[0]
            width = img.shape[1]
            N = (img[i][j + 1][k] - img[i][j][k]) if j + 1 < width else 0
            S = (img[i][j - 1][k] - img[i][j][k]) if j - 1 >= 0 else 0
            E = (img[i + 1][j][k] - img[i][j][k]) if i + 1 < height else 0
            W = (img[i - 1][j][k] - img[i][j][k]) if i - 1 >= 0 else 0
            return img[i][j][k] + 0.25 * (N * g(N, t) + S * g(S, t) + E * g(E, t) + W * g(W, t))

        for j in range(tmax):
            aux = FilterProvider.__anisotropic_matrix_filter(aux, anisotropic_single_filter, sigma, independent_layer)
        return aux

    @staticmethod
    def __leclerc(e, t):
        return np.exp(-((e / t) ** 2))

    @staticmethod
    def __lorentz(e, t):
        return 1 / ((e / t) ** 2 + 1)

    @staticmethod
    def __isotropic(e, t):
        return 1

    @staticmethod
    def __anisotropic_matrix_filter(matrix, func, sigma, independent_layer=False):
        ans = np.copy(matrix).astype(float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if independent_layer:
                    for k in range(matrix.shape[2]):
                        ans[i][j][k] = func(matrix, i, j, k, sigma)
                else:
                    aux = func(matrix, i, j, 0, sigma)
                    for k in range(matrix.shape[2]):
                        ans[i][j][k] = aux
        return ans

# img = Util.load_raw('LENA.RAW')
# img = FilterProvider.four_directions_border(img, merge_function=lambda p1, p2: p1 if p1 > p2 else p2)
# Util.save_raw(img, 'four_dir_borders_lena')

# img = Util.apply_to_matrix(img, lambda x: [x,x,x], two_dim=True)
# img_x = FilterProvider.x_border(img, False)
# img_x = Util.apply_to_matrix(img_x, lambda p: np.abs(p))
#
# img_y = FilterProvider.y_border(img, False)
# img_y = Util.apply_to_matrix(img_y, lambda p: np.abs(p))
# Util.save_raw(img_y, 'y_lena')

# img = Util.load_raw('../resources/test/LENA.RAW')
# salt_img = Util.add_comino_and_sugar_noise(img, density=0.1)
# Util.save_raw(salt_img, '../resources/test/salt_lena')
# aux = FilterProvider.anisotropic_filter(salt_img, 5, LORENTZ_BORDER_DETECTOR, 30)
# Util.save_raw(aux, '../resources/test/anisotropic_lena')
