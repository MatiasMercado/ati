import cv2
import numpy as np
import math
from src.input.filter_provider import FilterProvider
from src.input.provider import Provider
from src.input.util import Util
from matplotlib import pyplot as plt

class Harris:
    @staticmethod
    def harris_transformation(image,color,sigma,tp):
        Ix = np.zeros(image.shape)
        Iy = np.zeros(image.shape)
        Ixy = np.zeros(image.shape)
        cim1 = np.zeros(image.shape)
        cim2 = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1:
                        Ix[i][j][k] = image[i - 1][j + 1][k] + image[i][j + 1][k] + image[i + 1][j + 1][k] - (image[i - 1][j - 1][k] + image[i][j - 1][k] + image[i + 1][j - 1][k])
                        Iy[i][j][k] = image[i + 1][j - 1][k] + image[i + 1][j][k] + image[i + 1][j + 1][k] - (image[i - 1][j - 1][k] + image[i - 1][j][k] + image[i - 1][j + 1][k])
                        Ix[i][j][k] = Ix[i][j][k] ** 2
                        Iy[i][j][k] = Iy[i][j][k] ** 2
        gauss_filter_size = (2 * sigma + 1, 2 * sigma + 1)
        Ix = FilterProvider.gauss_blur(Ix, gauss_filter_size, sigma, color)
        Iy = FilterProvider.gauss_blur(Iy, gauss_filter_size, sigma, color)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    Ixy[i][j][k] = Ix[i][j][k] * Iy[i][j][k]
        Ixy = FilterProvider.gauss_blur(Ixy, gauss_filter_size, sigma, color)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1:
                        cim1[i][j][k] = (Ixy[i][j][k])- 0.04*((Ix[i][j][k]+Iy[i][j][k])**2)
                        if tp==0 and cim1[i][j][k] < 0:
                            image[i][j][1] = 255
                        elif tp==1 and cim1[i][j][k] > 0:
                            image[i][j][1] = 255
        return image

img = Util.load_image('cuadro.RAW')
img = Harris.harris_transformation(img,'false',2,1)
plt.imshow(img)
plt.show()
print (img)