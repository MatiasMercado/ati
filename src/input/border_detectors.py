from asyncio import Queue

import numpy as np

from src.input.filter_provider import FilterProvider
from src.input.provider import Provider
from src.input.util import Util

DEFAULT_ZERO_DETECTOR_THRESHOLD = 5

IN = -3
L_IN = -1
L_OUT = 1
OUT = 3


class BorderDetector:
    @staticmethod
    def laplacian_detector(image):
        mask = np.matrix([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]])
        aux_image = FilterProvider.sliding_window(image=image, mask=mask)
        aux_image = BorderDetector.__zero_detector(aux_image, threshold=4)
        return aux_image

    @staticmethod
    def laplacian_gaussian_detector(image, sigma):

        size = 7, 7
        mask = np.zeros(size)

        def gauss_function(x, y):
            return np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))) * (
                2 - ((x ** 2 + y ** 2) / (sigma ** 2))) / (-np.sqrt(2 * np.pi) * sigma ** 3)

        # Create Mask
        for i in range(size[0]):
            for j in range(size[1]):
                mask_index_i = i - int(mask.shape[0] / 2)  # Go from -rows/2 to +rows/2
                mask_index_j = j - int(mask.shape[1] / 2)  # Go from -cols/2 to +cols/2
                mask[i][j] = gauss_function(mask_index_i, mask_index_j)

        aux_image = FilterProvider.sliding_window(image=image, mask=mask)
        aux_image = BorderDetector.__zero_detector(aux_image, threshold=4)

        return aux_image

    @staticmethod
    def __zero_detector(image, independent_layer=False, threshold=DEFAULT_ZERO_DETECTOR_THRESHOLD):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]

        def single_zero_detector(p1, p2):
            if (p1 < 0 != p2 < 0) and np.abs(p1 - p2) > threshold:
                return 255
            return 0

        for x in range(image_width):
            for y in range(image_height):
                if independent_layer:
                    for z in range(image.shape[2]):
                        if x != image_width - 1:
                            ans[x, y, z] = single_zero_detector(image[x, y, z], image[x + 1, y, z])
                        if ans[x, y, z] != 255 and y != image_height - 1:
                            ans[x, y, z] = single_zero_detector(image[x, y, z], image[x, y + 1, z])
                else:
                    aux = 0
                    if x != image_width - 1:
                        aux = single_zero_detector(image[x, y, 0], image[x + 1, y, 0])
                    if aux == 0 and y != image_height - 1:
                        aux = single_zero_detector(image[x, y, 0], image[x, y + 1, 0])
                    for z in range(image.shape[2]):
                        ans[x, y, z] = aux
        return ans

    @staticmethod
    def global_threshold(image, threshold, delta, deltaT):
        if (deltaT < delta):
            return threshold
        else:
            ans = Util.to_binary(image, threshold)
            g1 = 0
            g2 = 0
            sumag1 = 0
            sumag2 = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if ans[i][j] == 0:
                        g1 += 1
                        sumag1 += image[i][j]
                    else:
                        g2 += 1
                        sumag2 += image[i][j]
            m1 = (1 / g1) * sumag1
            m2 = (1 / g2) * sumag2
            T = 0.5 * (m1 + m2)
            BorderDetector.global_threshold(image, T, delta, abs(T - threshold))
            return T

    @staticmethod
    def otsu_variable(hist, t, mg):
        p1 = 0
        mt = 0
        for i in range(t + 1):
            p1 = p1 + hist[i]
            mt = mt + (hist[i] * i)

        var = ((mg * p1 - mt) ** 2) / (p1 * (1 - p1))
        return var

    @staticmethod
    def otsu_threshold(image):
        hist = Provider.histogram(image)
        N = image.shape[0] * image.shape[1]
        mg = 0
        for i in range(256):
            hist[i] = hist[i] / N
            mg = mg + (hist[i] * i)
        vars = []
        for t in range(256):
            vars.append(BorderDetector.otsu_variable(hist, t, mg))
        tmax = max(vars)
        return vars.index(tmax)

    @staticmethod
    def generate_active_contour_initial_state(image, rectangle):
        v1, v2 = rectangle
        x1, y1 = v1
        x2, y2 = v2

        xi = min(x1, x2)
        yi = min(y1, y2)
        xf = max(x1, x2)
        yf = max(y1, y2)

        (width, height, depth) = image.shape

        aux = np.full((width, height), OUT, dtype=int)
        for x in range(width):
            for y in range(height):
                if xi < x < xf and yi < y < yf:
                    aux[x, y] = IN
                elif xi < x < xf:
                    if y == yi or y == yf:
                        aux[x, y] = L_IN
                        # print('l_in')
                    elif y == yi - 1 or y == yf + 1:
                        aux[x, y] = L_OUT
                        # print('l_out')
                elif yi < y < yf:
                    if x == xi or x == xf:
                        aux[x, y] = L_IN
                        # print('l_in')
                    elif x == xi - 1 or x == xf + 1:
                        aux[x, y] = L_OUT
                        # print('l_out')
        aux[xi, yi] = L_OUT
        aux[xf, yi] = L_OUT
        aux[xi, yf] = L_OUT
        aux[xf, yf] = L_OUT
        return aux

    @staticmethod
    def my_active_contour(image, initial_state, b_color, o_color, smooth=False):
        (width, height, depth) = image.shape
        print('shape:', image.shape)
        initial_l_in = Queue(maxsize=0)
        initial_l_out = Queue(maxsize=0)
        l_in = Queue(maxsize=0)
        l_out = Queue(maxsize=0)
        final_l_in = []
        final_l_out = []

        def calculate_difference(p):
            # print('calculate difference')
            b_diff = (p[0] - b_color[0]) ** 2
            b_diff += (p[1] - b_color[1]) ** 2
            b_diff += (p[2] - b_color[2]) ** 2
            o_diff = (p[0] - o_color[0]) ** 2
            o_diff += (p[1] - o_color[1]) ** 2
            o_diff += (p[2] - o_color[2]) ** 2
            return b_diff - o_diff

        def update_l_in_neighbors(p):
            if p[0] + 1 < initial_state.shape[0] and (initial_state[p[0] + 1, p[1]] == IN):
                initial_state[p[0] + 1, p[1]] = L_IN
                initial_l_in.put_nowait((p[0] + 1, p[1]))
            if p[0] - 1 >= 0 and (initial_state[p[0] - 1, p[1]] == IN):
                initial_state[p[0] - 1, p[1]] = L_IN
                initial_l_in.put_nowait((p[0] - 1, p[1]))
            if p[1] + 1 < initial_state.shape[1] and (initial_state[p[0], p[1] + 1] == IN):
                initial_state[p[0], p[1] + 1] = L_IN
                initial_l_in.put_nowait((p[0], p[1] + 1))
            if p[1] - 1 >= 0 and (initial_state[p[0], p[1] - 1] == IN):
                initial_state[p[0], p[1] - 1] = L_IN
                initial_l_in.put_nowait((p[0], p[1] - 1))

        def update_l_out_neighbors(p):
            if p[0] + 1 < initial_state.shape[0] and (initial_state[p[0] + 1, p[1]] == OUT):
                initial_state[p[0] + 1, p[1]] = L_OUT
                l_out.put_nowait((p[0] + 1, p[1]))
            if p[0] - 1 >= 0 and (initial_state[p[0] - 1, p[1]] == OUT):
                initial_state[p[0] - 1, p[1]] = L_OUT
                l_out.put_nowait((p[0] - 1, p[1]))
            if p[1] + 1 < initial_state.shape[1] and (initial_state[p[0], p[1] + 1] == OUT):
                initial_state[p[0], p[1] + 1] = L_OUT
                l_out.put_nowait((p[0], p[1] + 1))
            if p[1] - 1 >= 0 and (initial_state[p[0], p[1] - 1] == OUT):
                initial_state[p[0], p[1] - 1] = L_OUT
                l_out.put_nowait((p[0], p[1] - 1))

        def check_l_out_neighbors(p):
            if p[0] + 1 < initial_state.shape[0] and initial_state[p[0] + 1, p[1]] == L_IN:
                return True
            if p[0] - 1 >= 0 and initial_state[p[0] - 1, p[1]] == L_IN:
                return True
            if p[1] + 1 < initial_state.shape[1] and initial_state[p[0], p[1] + 1] == L_IN:
                return True
            if p[1] - 1 >= 0 and initial_state[p[0], p[1] - 1] == L_IN:
                return True
            return False

        def check_l_in_neighbors(p):
            if p[0] + 1 < initial_state.shape[0] and initial_state[p[0] + 1, p[1]] == L_OUT:
                return True
            if p[0] - 1 >= 0 and initial_state[p[0] - 1, p[1]] == L_OUT:
                return True
            if p[1] + 1 < initial_state.shape[1] and initial_state[p[0], p[1] + 1] == L_OUT:
                return True
            if p[1] - 1 >= 0 and initial_state[p[0], p[1] - 1] == L_OUT:
                return True
            return False

        for x in range(width):
            for y in range(height):
                if initial_state[x, y] == L_OUT:
                    initial_l_out.put_nowait((x, y))
                elif initial_state[x, y] == L_IN:
                    initial_l_in.put_nowait((x, y))

        while not initial_l_in.empty():
            x = initial_l_in.get_nowait()
            if calculate_difference(image[x[0], x[1]]) <= 0:
                initial_l_out.put_nowait(x)
                initial_state[x[0], x[1]] = L_OUT
                update_l_in_neighbors(x)
            else:
                l_in.put_nowait(x)

        while not initial_l_out.empty():
            x = initial_l_out.get_nowait()
            if check_l_out_neighbors(x):
                l_out.put_nowait(x)
            else:
                initial_state[x[0], x[1]] = OUT

        while not l_out.empty():
            x = l_out.get_nowait()
            if calculate_difference(image[x[0], x[1]]) > 0:
                l_in.put_nowait(x)
                initial_state[x[0], x[1]] = L_IN
                update_l_out_neighbors(x)
            else:
                final_l_out.append(x)

        while not l_in.empty():
            x = l_in.get_nowait()
            if check_l_in_neighbors(x):
                final_l_in.append(x)
            else:
                initial_state[x[0], x[1]] = IN

        if smooth:
            initial_state = BorderDetector.__smooth_contours(initial_state, final_l_in, final_l_out)

        return [initial_state, l_in, l_out]

    @staticmethod
    def __smooth_contours(state, l_in, l_out):
        initial_state = FilterProvider.gauss_blur(state, (5, 5), 1, two_dim=True)

        def check_gauss_l_in(p):
            if p[0] + 1 < initial_state.shape[0] and initial_state[p[0] + 1, p[1]] >= 0:
                return False
            if p[0] - 1 >= 0 and initial_state[p[0] - 1, p[1]] >= 0:
                return False
            if p[1] + 1 < initial_state.shape[1] and initial_state[p[0], p[1] + 1] >= 0:
                return False
            if p[1] - 1 >= 0 and initial_state[p[0], p[1] - 1] >= 0:
                return False
            return True

        def check_gauss_l_out(p):
            if p[0] + 1 < initial_state.shape[0] and initial_state[p[0] + 1, p[1]] < 0:
                return False
            if p[0] - 1 >= 0 and initial_state[p[0] - 1, p[1]] < 0:
                return False
            if p[1] + 1 < initial_state.shape[1] and initial_state[p[0], p[1] + 1] < 0:
                return False
            if p[1] - 1 >= 0 and initial_state[p[0], p[1] - 1] < 0:
                return False
            return True

        for x in l_out:
            if initial_state[x[0], x[1]] < 0:
                l_out.remove(x)
                l_in.append(x)
                initial_state[x[0], x[1]] = L_IN
        for x in l_in:
            if check_gauss_l_in(x):
                l_in.remove(x)
                initial_state[x[0], x[1]] = IN
        for x in l_in:
            if initial_state[x[0], x[1]] >= 0:
                l_out.append(x)
                l_in.remove(x)
                initial_state[x[0], x[1]] = L_OUT
        for x in l_out:
            if check_gauss_l_out(x):
                l_out.remove(x)
                initial_state[x[0], x[1]] = OUT

        return initial_state

    @staticmethod
    def active_contour(image, initial_state, b_color, o_color, smooth=False, max_iterations=1000):
        (width, height, depth) = image.shape
        print('shape:', image.shape)
        l_in = []
        l_out = []

        def calculate_difference(p):
            # print('calculate difference')
            b_diff = (p[0] - b_color[0]) ** 2
            b_diff += (p[1] - b_color[1]) ** 2
            b_diff += (p[2] - b_color[2]) ** 2
            o_diff = (p[0] - o_color[0]) ** 2
            o_diff += (p[1] - o_color[1]) ** 2
            o_diff += (p[2] - o_color[2]) ** 2
            return b_diff - o_diff

        def update_l_in(p):

            n = initial_state[p[0] + 1, p[1]]
            if p[0] + 1 < initial_state.shape[0] and (n == OUT or n == L_OUT):
                return False
            n = initial_state[p[0] - 1, p[1]]
            if p[0] - 1 >= 0 and (n == OUT or n == L_OUT):
                return False
            n = initial_state[p[0], p[1] + 1]
            if p[1] + 1 < initial_state.shape[1] and (n == OUT or n == L_OUT):
                return False
            n = initial_state[p[0], p[1] - 1]
            if p[1] - 1 >= 0 and (n == OUT or n == L_OUT):
                return False
            return True

        def update_l_out(p):
            n = initial_state[p[0] + 1, p[1]]
            if p[0] + 1 < initial_state.shape[0] and (n == IN or n == L_IN):
                return False
            n = initial_state[p[0] - 1, p[1]]
            if p[0] - 1 >= 0 and (n == IN or n == L_IN):
                return False
            n = initial_state[p[0], p[1] + 1]
            if p[1] + 1 < initial_state.shape[1] and (n == IN or n == L_IN):
                return False
            n = initial_state[p[0], p[1] - 1]
            if p[1] - 1 >= 0 and (n == IN or n == L_IN):
                return False
            return True

        def update_l_in_neighbors(p):
            if p[0] + 1 < initial_state.shape[0] and (initial_state[p[0] + 1, p[1]] == IN):
                initial_state[p[0] + 1, p[1]] = L_IN
                l_in.append((p[0] + 1, p[1]))
            if p[0] - 1 >= 0 and (initial_state[p[0] - 1, p[1]] == IN):
                initial_state[p[0] - 1, p[1]] = L_IN
                l_in.append((p[0] - 1, p[1]))
            if p[1] + 1 < initial_state.shape[1] and (initial_state[p[0], p[1] + 1] == IN):
                initial_state[p[0], p[1] + 1] = L_IN
                l_in.append((p[0], p[1] + 1))
            if p[1] - 1 >= 0 and (initial_state[p[0], p[1] - 1] == IN):
                initial_state[p[0], p[1] - 1] = L_IN
                l_in.append((p[0], p[1] - 1))

        def update_l_out_neighbors(p):
            if p[0] + 1 < initial_state.shape[0] and (initial_state[p[0] + 1, p[1]] == OUT):
                initial_state[p[0] + 1, p[1]] = L_OUT
                l_out.append((p[0] + 1, p[1]))
            if p[0] - 1 >= 0 and (initial_state[p[0] - 1, p[1]] == OUT):
                initial_state[p[0] - 1, p[1]] = L_OUT
                l_out.append((p[0] - 1, p[1]))
            if p[1] + 1 < initial_state.shape[1] and (initial_state[p[0], p[1] + 1] == OUT):
                initial_state[p[0], p[1] + 1] = L_OUT
                l_out.append((p[0], p[1] + 1))
            if p[1] - 1 >= 0 and (initial_state[p[0], p[1] - 1] == OUT):
                initial_state[p[0], p[1] - 1] = L_OUT
                l_out.append((p[0], p[1] - 1))

        for x in range(width):
            for y in range(height):
                if initial_state[x, y] == L_OUT:
                    l_out.append((x, y))
                elif initial_state[x, y] == L_IN:
                    l_in.append((x, y))
        i = 1
        touched = True
        while touched and i < max_iterations:
            touched = False
            print('it:', i)
            i += 1
            for x in l_out:
                if calculate_difference(image[x[0], x[1]]) > 0:
                    touched = True
                    l_out.remove(x)
                    l_in.append(x)
                    initial_state[x[0], x[1]] = L_IN
                    update_l_out_neighbors(x)
            for x in l_in:
                if update_l_in(x):
                    initial_state[x[0], x[1]] = IN
                    l_in.remove((x[0], x[1]))
            for x in l_in:
                if calculate_difference(image[x[0], x[1]]) <= 0:
                    touched = True
                    l_in.remove(x)
                    l_out.append(x)
                    update_l_in_neighbors(x)
            for x in l_out:
                if update_l_out(x):
                    initial_state[x[0], x[1]] = OUT
                    l_out.remove((x[0], x[1]))

        if smooth:
            initial_state = BorderDetector.__smooth_contours(initial_state, l_in, l_out)

        return [initial_state, l_in, l_out]

# img = Util.load_image('mate.jpg')
# state = BorderDetector.generate_active_contour_initial_state(img, ((115, 155), (138, 165)))
# state_mine = BorderDetector.generate_active_contour_initial_state(img, ((115, 155), (138, 165)))
# start = time.time()
# result, l_in, l_out = BorderDetector.active_contour(img, state, [255, 255, 255], [200, 87, 42], smooth=True)
# end_original = time.time()
# result_mine, l_in_mine, l_out_mine = BorderDetector.my_active_contour(img, state, [255, 255, 255], [200, 87, 42], smooth=True)
# end_mine = time.time()
# print('TIMES')
# print('original algorithm:', end_original - start)
# print('mine:', end_mine - end_original)
# print('ratio:', (end_original - start) / (end_mine - end_original))
#
# result = Util.to_binary(result, 0)
# result_min = Util.to_binary(result_mine, 0)
#
# for p in l_in:
#     img[p[0], p[1]] = [255, 255, 45]
#
# img_to_save = Image.fromarray(result.astype(np.uint8))
# img_to_save.save('active.png')
# img_to_save = Image.fromarray(result_mine.astype(np.uint8))
# img_to_save.save('active_mine.png')
# img_to_save = Image.fromarray(img.astype(np.uint8))
# img_to_save.save('on_image_active.png')
