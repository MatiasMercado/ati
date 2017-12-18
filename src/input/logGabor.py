import math

import cv2
import numpy as np

from src.input.provider import Provider
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
        degrees = (246 - 180 - 1) + (359 - 314 - 1)
        ans = np.zeros((36, degrees))  # Resize al tamaño de interes donde hay menos ruido
        for i in range(normalized_iris.shape[0]):
            for j in range(normalized_iris.shape[1]):
                if 180 < j < 246:
                    ans[i][j - 181] = normalized_iris[i][j]
                elif 314 < j < 359:
                    ans[i][j - 315 + 246 - 180 - 1] = normalized_iris[i][j]
        return ans

    @staticmethod
    def build_filters(ksize=27):
        filters = []
        print('kernels:')
        for theta in np.arange(0, np.pi, np.pi / 4):
            for f in [2, 4, 8, 16, 32, 64]:
                kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, f, 0.5, 0, ktype=cv2.CV_32F)
                # kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    @staticmethod
    def process(img, filters):
        features = []
        for kern in filters:
            filteredImage = cv2.filter2D(img, cv2.CV_8UC3, kern)
            mean = np.mean(filteredImage)
            std = np.std(filteredImage)
            features.append((mean, std))
            # np.maximum(template, fimg, template)
        return features

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
    def compare_templates_euclidean(t1, t2):
        acu = 0
        width, height = t1.shape
        for i in range(width):
            for j in range(height):
                acu += (t1[i][j] - t2[i][j]) ** 4
        return np.sqrt(acu)

    @staticmethod
    def compare_templates_threshold(t1, t2, threshold=25):
        acu = 0
        width, height = t1.shape
        total = width * height
        for i in range(width):
            for j in range(height):
                if np.abs(t1[i][j] - t2[i][j]) <= threshold:
                    acu += 1
        return acu / total

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

    @staticmethod
    def compare_templates_w_euclidean(f1, f2, name1='image1', name2='image2', threshold=0.7):
        acu = 0
        for i in range(len(f1)):
            if f1[i][1] != 0:
                acu += ((f1[i][0] - f2[i][0]) ** 2) / (f1[i][1] ** 2)
        print('comparison result:')
        print(acu)
        with open("log.txt", "a") as log_file:
            log_file.write(str(name1) + ' - ' + str(name2) + ' => ' + str(acu))
        return acu < threshold


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

# diana = Provider.draw_diana()
# diana_normalized = LogGabor.normalization(diana, , )
# cv2.imwrite('/home/mati/Documents/pythonenv/ati/src/input/diana.jpg', diana_normalized)

def compare(name1, name2, ksize=27):
    filters = LogGabor.build_filters(ksize)

    template1 = Util.load_image('/home/mati/Documents/pythonenv/ati/src/input/result/snipped_' + str(name1) + '.jpg')
    # template1 = Util.load_image('/Users/jcl/PycharmProjects/ati/ati/src/input/result/snipped_' + str(name1) + '.jpg')
    template1 = cv2.cvtColor(template1.astype('B'), cv2.COLOR_BGR2GRAY)
    template1 = Provider.equalize_histogram(template1, two_dim=True)
    features1 = LogGabor.process(template1, filters)

    template2 = Util.load_image('/home/mati/Documents/pythonenv/ati/src/input/result/snipped_' + str(name2) + '.jpg')
    # template2 = Util.load_image('/Users/jcl/PycharmProjects/ati/ati/src/input/result/snipped_' + str(name2) + '.jpg')
    template2 = cv2.cvtColor(template2.astype('B'), cv2.COLOR_BGR2GRAY)
    template2 = Provider.equalize_histogram(template2, two_dim=True)
    features2 = LogGabor.process(template2, filters)

    result = LogGabor.compare_templates_w_euclidean(features1, features2)
    print('Comparison:')
    print(result)
    return result
    # print('euclidean')
    # print(LogGabor.compare_templates_euclidean(template1, template2))
    # print('threshold')
    # print(LogGabor.compare_templates_threshold(template1, template2, 1))


# results = []
#
# for ks in range(8, 40):
#     print(ks)
#     c1 = compare(1, 2, ks)
#     c2 = compare(1, 'philip', ks)
#     c3 = compare(2, 'philip', ks)
#     results.append([c2 * c3 / (c1 * 2), ks, c1, c2, c3])
# print(results)
# print(max(results))
