import numpy as np
import math

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
    def prewitt_edge(image):
        ans = np.zeros(image.shape)
        fi=np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if (i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1):
                        dx = image[i - 1][j + 1][k] + image[i    ][j + 1][k] + image[i + 1][j + 1][k] - (image[i - 1][j - 1][k] + image[i    ][j - 1][k] + image[i + 1][j - 1][k])
                        dy = image[i + 1][j - 1][k] + image[i + 1][j    ][k] + image[i + 1][j + 1][k] - (image[i - 1][j - 1][k] + image[i - 1][j    ][k] + image[i - 1][j + 1][k])
                        ans[i][j][k]=math.sqrt(dx**2+dy**2)
                        angle=math.degrees(math.atan2(dy,dx))
                        if(angle<0):
                            angle=angle+180
                        if(angle<22.5 or angle>157.5):
                            fi[i][j][k]=0
                        elif(angle>22.5 and angle<67.5):
                            fi[i][j][k] = 45
                        elif (angle>67.5 and angle<112.5):
                            fi[i][j][k] = 90
                        elif (angle>112.5 and angle<157.5):
                            fi[i][j][k] = 135
        return (Util.linear_transform(ans),fi)

    @staticmethod
    def no_maximos(image,fi):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if (i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1):
                        pix=image[i][j][k]
                        if(fi[i][j][k]==0):
                            if(image[i  ][j-1][k]> pix or image[i  ][j+1][k]> pix):
                                image[i][j][k]=0
                        elif (fi[i][j][k] == 45):
                            if (image[i-1][j-1][k] > pix or image[i+1][j+1][k] > pix):
                                image[i][j][k] = 0
                        elif (fi[i][j][k] == 90):
                            if (image[i - 1][j][k] > pix or image[i + 1][j][k] > pix):
                                image[i][j][k] = 0
                        elif (fi[i][j][k] == 135):
                            if (image[i+1][j-1][k] > pix or image[i-1][j+1][k] > pix):
                                image[i][j][k] = 0
        return image

    @staticmethod
    def hysteresis(image, t1, t2):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if (i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1):
                        if(image[i][j][k]<t1):
                            image[i][j][k]=0
                        elif(image[i][j][k]>t2):
                            image[i][j][k]=255
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
        aux= BorderDetector.hysteresis(aux,t-(desv/2),t+(desv/2))
        aux1= BorderDetector.hysteresis(aux1,t-(desv/2),t+(desv/2))
        ans =np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    ans[i][j][k] = aux1[i][j][k]+aux[i][j][k]
        return ans

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
