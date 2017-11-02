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
    def hough_transform(image, theta_steps, p_steps, epsilon=0.9, number=True):
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

'''
img = Util.load_raw('LENA.RAW')
ans=np.zeros(img.shape)
t=BorderDetector.global_threshold(img,ans,100,1,200)
img= Util.to_binary(img,t)
print(img)
plt.imshow(img)
plt.show()    

img=Provider.histogram(img)
print(img)
plt.hist(img)
plt.show()

ans=np.zeros(img.shape)
img = BorderDetector.global_threshold(img,ans,55,5,255)
plt.imshow(img)
plt.show()    '''

# img = BorderDetector.laplacian_gaussian_detector(img, 0.7)
# Util.save_raw(img, 'lena_laplacian_gaussian_border')
