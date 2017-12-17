import math

import cv2
import numpy as np

from input.provider import Provider
from src.input.util import Util


class LogGabor:

    @staticmethod
    def normalization(image, inner, outer):
        iris = np.zeros((36, 360))

        innerimax = max(inner)
        innerimin = min(inner)
        # center = (int((innerimax[1] + innerimin[1]) / 2), int((innerimax[0] + innerimin[0]) / 2))
        center = (int((innerimax[0] + innerimin[0]) / 2), int((innerimax[1] + innerimin[1]) / 2))
        # innerradius = int(math.sqrt((center[0] - inner[0][1]) ** 2 + (center[1] - inner[0][0]) ** 2))
        innerradius = int(math.sqrt((center[0] - inner[0][0]) ** 2 + (center[1] - inner[0][1]) ** 2))
        # outterradius = int(math.sqrt((center[0] - outer[0][1]) ** 2 + (center[1] - outer[0][0]) ** 2))
        outterradius = int(math.sqrt((center[0] - outer[0][0]) ** 2 + (center[1] - outer[0][1]) ** 2))

        iris_height = outterradius - innerradius

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pointradius = int(math.sqrt((center[0] - i) ** 2 + (center[1] - j) ** 2))
                if innerradius <= pointradius < outterradius:
                    theta = int(math.degrees(math.atan2(j - center[1], i - center[0])))
                    if theta < 0:
                        theta += 360
                    iris[int((pointradius - innerradius) * 35 / iris_height)][theta] = image[i][j]
                    if theta < 359:
                        iris[int((pointradius - innerradius) * 35 / iris_height)][theta + 1] = image[i][j]
                        if theta < 358:
                            iris[int((pointradius - innerradius) * 35 / iris_height)][theta + 2] = image[i][j]
        return iris

    @staticmethod
    def interest_degrees(normalized_iris):
        ans = np.zeros((36, 360))  # Resize al tamaño de interes donde hay menos ruido
        for i in range(normalized_iris.shape[0]):
            for j in range(normalized_iris.shape[1]):
                if 180 < j < 246 or 314 < j < 359:
                    ans[i][j] = normalized_iris[i][j]
        return ans

    @staticmethod
    def build_filters():
        filters = []
        ksize = 7
        print('kernels:')
        for theta in np.arange(0, np.pi, np.pi / 8):
            # print(math.degrees(theta))
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            print(kern)
            kern /= 1.5 * kern.sum()
            print(kern)
            filters.append(kern)
        return filters

    @staticmethod
    def process(img, filters):
        print('img')
        print(img)
        template = np.zeros_like(img)
        print('length of builders:')
        print(len(filters))
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(template, fimg, template)
        return template

    @staticmethod
    def ceropercent(template):
        acumm = 0
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                if template[i][j] == 1:
                    acumm += 1
        print(acumm / (template.shape[0] * template.shape[1]))
        return acumm

    @staticmethod
    def compare_templates(basetemplate, template):
        acumm = 0
        total = 0
        for i in range(basetemplate.shape[0]):
            for j in range(basetemplate.shape[1]):
                comp = LogGabor.__compare_neighbors((i, j), (i, j), basetemplate, template)
                if comp[0]:
                    total += 1
                if comp[1]:
                    acumm += 1
        hammingdist = acumm / total
        return hammingdist

    @staticmethod
    def __compare_neighbors(p1, p2, t1, t2, neighborhood=0):
        p1_value = False
        p2_value = False
        for i in range(np.max([0, p1[0] - neighborhood]), np.min([t1.shape[0] - 1, p1[0] + neighborhood]) + 1):
            for j in range(np.max([0, p1[1] - neighborhood]), np.min([t1.shape[1] - 1, p1[1] + neighborhood]) + 1):
                if t1[i][j] != 0:
                    p1_value = True
                    break
        for i in range(np.max([0, p2[0] - neighborhood]), np.min([t2.shape[0] - 1, p2[0] + neighborhood]) + 1):
            for j in range(np.max([0, p2[1] - neighborhood]), np.min([t2.shape[1] - 1, p2[1] + neighborhood]) + 1):
                if t2[i][j] != 0:
                    p2_value = True
                    break
        return p1_value or p2_value, p1_value and p2_value


# image = Util.load_image('src/input/LENA.RAW')
# image = Util.load_image('/Users/jcl/PycharmProjects/ati/ati/src/input/result/iris.jpg')
# image = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
# filters = LogGabor.build_filters()
#
# image = Util.load_image('/Users/jcl/PycharmProjects/ati/ati/src/input/result/iris.jpg')


# image = Util.load_image('/home/mati/Documents/pythonenv/ati/src/input/result/iris_philip.jpg')
# image = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
# filters = LogGabor.build_filters()
# template = LogGabor.process(image, filters)
# cv2.imwrite('/home/mati/Documents/pythonenv/ati/src/input/iris_philip_gabor.jpg', template)
diana = Provider.draw_diana()
cv2.imwrite('/home/mati/Documents/pythonenv/ati/src/input/diana.jpg', diana)

def compare(name1, name2):
    template1 = Util.load_image('/Users/jcl/PycharmProjects/ati/ati/src/input/result/snipped_' + str(name1) + '.jpg')
    template1 = cv2.cvtColor(template1.astype('B'), cv2.COLOR_BGR2GRAY)
    template2 = Util.load_image('/Users/jcl/PycharmProjects/ati/ati/src/input/result/snipped_' + str(name2) + '.jpg')
    template2 = cv2.cvtColor(template2.astype('B'), cv2.COLOR_BGR2GRAY)

    print(LogGabor.compare_templates(template1, template2))

#
# compare(4, 5)
# compare(4, 'philip')
