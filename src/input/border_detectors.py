import math
from asyncio import Queue

import numpy as np
import queue as q
from src.input.filter_provider import FilterProvider
from src.input.provider import Provider
from src.input.util import Util

SUSAN_BORDER_DETECTOR = 0
SUSAN_CORNER_DETECTOR = 1
SUSAN_BORDER_CORNER_DETECTOR = 2
SUSAN_BORDER_POINT = 1
SUSAN_CORNER_POINT = 2

DEFAULT_ZERO_DETECTOR_THRESHOLD = 5

IN = -3
L_IN = -1
L_OUT = 1
OUT = 3


class BorderDetector:
    @staticmethod
    def laplacian_detector(image, threshold=4, independent_layer=False):
        mask = np.matrix([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]])
        aux_image = FilterProvider.sliding_window(image=image, mask=mask, independent_layer=independent_layer)
        aux_image = BorderDetector.__zero_detector(aux_image, independent_layer, threshold)
        return aux_image

    @staticmethod
    def laplacian_gaussian_detector(image, sigma, threshold, independent_layer=False):

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

        aux_image = FilterProvider.sliding_window(image=image, mask=mask, independent_layer=independent_layer)
        aux_image = BorderDetector.__zero_detector(aux_image, independent_layer, threshold)

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
        if deltaT < delta:
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
    def susan_border_detector(image, independent_layer=False, detector_type=SUSAN_BORDER_CORNER_DETECTOR,
                              delta=0.1):
        (image_width, image_height) = image.shape[0], image.shape[1]
        ans = np.copy(image)
        border_pixels = FilterProvider.susan_sliding_window(image, independent_layer, detector_type, delta)
        for x in range(image_width):
            for y in range(image_height):
                if border_pixels[x, y, 0] == SUSAN_BORDER_POINT:
                    ans[x, y, 0] = 0
                    ans[x, y, 1] = 255
                    ans[x, y, 2] = 0
                elif border_pixels[x, y, 0] == SUSAN_CORNER_POINT:
                    ans[x, y, 0] = 255
                    ans[x, y, 1] = 0
                    ans[x, y, 2] = 0
        return ans

    @staticmethod
    def hough_transform(image, theta_steps, p_steps, epsilon=0.9, number=False):
        (image_width, image_height) = image.shape[0], image.shape[1]
        D = max(image_width, image_height)

        # Theta Interval [-90,90] degrees in step of 1 degree equals ~ [-1.5, 1.5] step 0.01 in rad
        theta_start = -90 * np.pi / 180
        theta_end = 90 * np.pi / 180
        p_start = - D * np.sqrt(2)
        p_end = D * np.sqrt(2)

        if number: # steps indicate the number of steps
            theta_step = (theta_end - theta_start) / theta_steps
            p_step = (p_end - p_start) / p_steps
        else: # steps indicate the value of the step
            theta_step = theta_steps * np.pi / 180
            p_step = p_steps

        theta_range = np.arange(theta_start, theta_end, theta_step)
        p_range = np.arange(p_start, p_end, p_step)
        lines = np.zeros((theta_range.size, p_range.size))
        points = {}
        for x in range(image_height):
            for y in range(image_width):
                if image[x, y, 0] == 255:
                    for a in range(theta_range.size):
                        theta = theta_range[a]
                        for b in range(p_range.size):
                            p = p_range[b]
                            if np.abs(p - x * np.cos(theta) - y * np.sin(theta)) < epsilon:
                                lines[a,b] += 1
                                if lines[a, b] == 1:
                                    points[(a,b)] = []
                                points[(a,b)].append((x,y))
        return lines, points, theta_range, p_range

    @staticmethod
    def prewitt_edge(image):
        ans = np.zeros(image.shape)
        fi = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1:
                        dx = image[i - 1][j + 1][k] + image[i][j + 1][k] + image[i + 1][j + 1][k] - (
                            image[i - 1][j - 1][k] + image[i][j - 1][k] + image[i + 1][j - 1][k])
                        dy = image[i + 1][j - 1][k] + image[i + 1][j][k] + image[i + 1][j + 1][k] - (
                            image[i - 1][j - 1][k] + image[i - 1][j][k] + image[i - 1][j + 1][k])
                        ans[i][j][k] = math.sqrt(dx ** 2 + dy ** 2)
                        angle = math.degrees(math.atan2(dy, dx))
                        if angle < 0:
                            angle = angle + 180
                        if angle < 22.5 or angle > 157.5:
                            fi[i][j][k] = 0
                        elif angle > 22.5 and angle < 67.5:
                            fi[i][j][k] = 45
                        elif angle > 67.5 and angle < 112.5:
                            fi[i][j][k] = 90
                        elif angle > 112.5 and angle < 157.5:
                            fi[i][j][k] = 135
        return Util.linear_transform(ans), fi

    @staticmethod
    def no_maximos(image, fi):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1:
                        pix = image[i][j][k]
                        if fi[i][j][k] == 0:
                            if image[i][j - 1][k] > pix or image[i][j + 1][k] > pix:
                                image[i][j][k] = 0
                        elif fi[i][j][k] == 45:
                            if image[i - 1][j - 1][k] > pix or image[i + 1][j + 1][k] > pix:
                                image[i][j][k] = 0
                        elif fi[i][j][k] == 90:
                            if image[i - 1][j][k] > pix or image[i + 1][j][k] > pix:
                                image[i][j][k] = 0
                        elif fi[i][j][k] == 135:
                            if image[i + 1][j - 1][k] > pix or image[i - 1][j + 1][k] > pix:
                                image[i][j][k] = 0
        return image

@staticmethod
def hysteresis(image, t1, t2):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1:
                        if image[i][j][k] < t1:
                            image[i][j][k] = 0
                        if image[i][j][k] > t2:
                            image[i][j][k] = 255
                        if image[i-1][j][k] > t2 or image[i][j-1][k] > t2 or image[i+1][j][k] > t2 or image[i][j+1][k] > t2 or image[i-1][j-1][k] > t2 or image[i+1][j+1][k] > t2 or image[i-1][j+1][k] > t2 or image[i+1][j-1][k] > t2:
                        	image[i][j][k] == 255
                        else:
                        	image[i][j][k] == 0
        return image

    @staticmethod
    def desv(image):
        ac = 0
        n = image.shape[0] * image.shape[1] * image.shape[2]
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    ac += image[i][j][k]
        media = ac / n
        ac = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    ac += (image[i][j][k] - media) ** 2
        desv = math.sqrt(ac / (n - 1))
        return desv

    @staticmethod
    def canny_edges(sigma, sigma2, image, color):
        gauss_filter_size1 = (2 * sigma + 1, 2 * sigma + 1)
        aux = FilterProvider.gauss_blur(image, gauss_filter_size1, sigma, color)
        (aux, fi) = BorderDetector.prewitt_edge(aux)
        aux = BorderDetector.no_maximos(aux, fi)

        gauss_filter_size2 = (2 * sigma2 + 1, 2 * sigma2 + 1)
        aux1 = FilterProvider.gauss_blur(image, gauss_filter_size2, sigma2, color)
        (aux1, fi) = BorderDetector.prewitt_edge(aux1)
        aux1 = BorderDetector.no_maximos(aux1, fi)

        t = BorderDetector.otsu_threshold(image)
        desv = BorderDetector.desv(image)
        aux = BorderDetector.hysteresis(aux, t - (desv / 2), t + (desv / 2))
        aux1 = BorderDetector.hysteresis(aux1, t - (desv / 2), t + (desv / 2))
        ans = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if aux1[i][j][k] == 255 or aux[i][j][k] == 255:
                        ans[i][j][k] = 255
                    else:
                        ans[i][j][k] = 0

        return ans

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
    def my_active_contour(image, initial_state, b_color, o_color, smooth=False, borders=([], [])):
        (width, height, depth) = image.shape
        print('shape:', image.shape)

        initial_l_in = Queue(maxsize=0)
        initial_l_out = Queue(maxsize=0)
        l_in = Queue(maxsize=0)
        l_out = Queue(maxsize=0)
        final_l_in = []
        final_l_out = []

        for x in borders[0]:
            initial_l_in.put_nowait(x)
        for x in borders[1]:
            initial_l_out.put_nowait(x)

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

        if initial_l_in.empty():
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

        return [initial_state, final_l_in, final_l_out]

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
    def active_contour(image, initial_state, b_color, o_color, smooth=False, max_iterations=1000, borders=([], [])):
        print('shape:', image.shape)
        (width, height, depth) = image.shape
        l_in, l_out = borders

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

            if p[0] + 1 < initial_state.shape[0]:
                n = initial_state[p[0] + 1, p[1]]
                if n == OUT or n == L_OUT:
                    return False
            if p[0] - 1 >= 0:
                n = initial_state[p[0] - 1, p[1]]
                if n == OUT or n == L_OUT:
                    return False
            if p[1] + 1 < initial_state.shape[1]:
                n = initial_state[p[0], p[1] + 1]
                if n == OUT or n == L_OUT:
                    return False
            if p[1] - 1 >= 0:
                n = initial_state[p[0], p[1] - 1]
                if n == OUT or n == L_OUT:
                    return False
            return True

        def update_l_out(p):
            if p[0] + 1 < initial_state.shape[0]:
                n = initial_state[p[0] + 1, p[1]]
                if n == IN or n == L_IN:
                    return False
            if p[0] - 1 >= 0:
                n = initial_state[p[0] - 1, p[1]]
                if n == IN or n == L_IN:
                    return False
            if p[1] + 1 < initial_state.shape[1]:
                n = initial_state[p[0], p[1] + 1]
                if n == IN or n == L_IN:
                    return False
            if p[1] - 1 >= 0:
                n = initial_state[p[0], p[1] - 1]
                if n == IN or n == L_IN:
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

        if len(l_in) == 0:
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

    @staticmethod
    def active_contour_sequence(image_array, initial_state, b_color, o_color, max_iterations=1000, algorithm=0):
        result_array = []
        l_in = []
        l_out = []
        line_color = [255, 255, 45]
        result = initial_state
        for image in image_array:
            borders = (l_in, l_out)
            if algorithm == 0:
                result, l_in, l_out = BorderDetector.active_contour(image, result, b_color, o_color,
                                                                    max_iterations=max_iterations, borders=borders)
            else:
                result, l_in, l_out = BorderDetector.my_active_contour(image, result, b_color, o_color, borders=borders)
            for p in l_in:
                image[p[0], p[1]] = line_color

            result_array.append(image)
        return result_array


# img = Util.load_image('mate.jpg')
##images = [Util.load_image('mate.jpg'), Util.load_image('mate.jpg'), Util.load_image('mate.jpg')]
##state = BorderDetector.generate_active_contour_initial_state(images[0], ((115, 155), (138, 165)))
# state_mine = BorderDetector.generate_active_contour_initial_state(img, ((115, 155), (138, 165)))
# start = time.time()
##results = BorderDetector.active_contour_sequence(images, state, [255, 255, 255], [200, 87, 42])
# result, l_in, l_out = BorderDetector.active_contour(img, state, [255, 255, 255], [200, 87, 42], smooth=True)
# end_original = time.time()
# result_mine, l_in_mine, l_out_mine = BorderDetector.my_active_contour(img, state, [255, 255, 255], [200, 87, 42], smooth=True)
# end_mine = time.time()
# print('TIMES')
# print('original algorithm:', end_original - start)
# print('mine:', end_mine - end_original)
# print('ratio:', (end_original - start) / (end_mine - end_original))

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

## image_count = 0
## for result in results:
##     img_to_save = Image.fromarray(result.astype(np.uint8))
##     img_to_save.save('on_image_active_' + str(image_count) + '.png')
##     image_count += 1
