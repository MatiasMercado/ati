import numpy as np

from src.input.filter_provider import FilterProvider
from src.input.provider import Provider
from src.input.util import Util

DEFAULT_ZERO_DETECTOR_THRESHOLD = 5


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
    def global_threshold(image, ans, threshold, delta, deltaT):
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
                    for k in range(image.shape[2]):
                        if ans[i][j][k] == 0:
                            g1 += 1
                            sumag1 += image[i][j][k]
                        else:
                            g2 += 1
                            sumag2 += image[i][j][k]
            m1 = (1 / g1) * sumag1
            m2 = (1 / g2) * sumag2
            T = 0.5 * (m1 + m2)
            BorderDetector.global_threshold(image, ans, T, delta, abs(T - threshold))
            return T

    @staticmethod
    def otsu_variable(hist, t, mg):
        p1 = 0
        mt = 0
        # for i in range(0, t + 1):
        #     p1 = p1 + hist[i]
        # for i in range(0, t + 1):
        #     mt = mt + (hist[i] * i)
        # for i in range(256):
        #     mg = mg + (hist[i] * i)
        #
        for i in range(256):
            if i <= t:
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
        for i in range(len(vars)):
            if vars[i] > 255:
                vars[i] = 0
        tmax = max(vars)
        # print(vars)
        return tmax


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
