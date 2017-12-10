import numpy as np
from cv2 import cv2

from src.input.filter_provider import FilterProvider
from src.input.vector_util import VectorUtil


class FeaturesDetector:
    @staticmethod
    def SIFT(img1, img2):
        gray1 = cv2.cvtColor(img1.astype('B'), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2.astype('B'), cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        img1 = cv2.drawKeypoints(gray1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2 = cv2.drawKeypoints(gray2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        print('matches:')
        print(matches)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = np.zeros(img1.shape)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, flags=2)

        print(len(good) / len(matches))
        print('keypoints1: ' + str(len(kp1)))
        print('keypoints2: ' + str(len(kp2)))
        print('good: ' + str(len(good)))

        return img3

    @staticmethod
    def SIFT_single(img1):
        gray1 = cv2.cvtColor(img1.astype('B'), cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(gray1, None)

        img1 = cv2.drawKeypoints(gray1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img1

    @staticmethod
    def curvature_energy(control_point, prev_point, next_point):
        x, y = control_point
        p1, p2 = prev_point
        n1, n2 = next_point
        return (p1 - 2 * x + n1) ** 2 + (p2 - 2 * y + n2) ** 2

    @staticmethod
    def continuity_energy(control_point, average_distance, prev_point):
        return average_distance - np.sqrt(VectorUtil.sqr_euclidean_distance(control_point, prev_point))

    @staticmethod
    def average_distance(control_points):
        # Set last point as previous for the first element
        prev = control_points[len(control_points) - 1]
        average = 0
        for point in control_points:
            average += np.sqrt(VectorUtil.sqr_euclidean_distance(point, prev))
            prev = point
        return average / len(control_points)

    @staticmethod
    def second_curvature_energy(control_point, prev_point, next_point):
        x, y = control_point
        p1, p2 = prev_point
        n1, n2 = next_point

        hi = (x - p1, y - p2)
        hi1 = (n1 - x, n2 - y)

        hi_abs = VectorUtil.vector_abs(hi)
        hi1_abs = VectorUtil.vector_abs(hi1)

        hi_normilzed = (hi[0] / hi_abs, hi[1] / hi_abs)
        hi1_normilzed = (hi1[0] / hi1_abs, hi1[1] / hi1_abs)

        return VectorUtil.vector_abs((hi1_normilzed[0] - hi_normilzed[0],
                                      hi1_normilzed[1] - hi_normilzed[1]))

    @staticmethod
    def image_energy(image, position, direction):
        return FilterProvider.single_point_gradient(image, position, direction, weighted=True)

    @staticmethod
    def iris_detector(image, initial_state, alpha=0.5, beta=0.5, gamma=0.5, iterations=10):
        length = len(initial_state)
        for i in range(iterations):
            for index in range(length):
                initial_state[index] = FeaturesDetector.find_lowest_energy(
                    image, initial_state[index], initial_state[(index + 1) % length],
                    initial_state[(index - 1) % length], alpha, beta, gamma,
                    FeaturesDetector.average_distance(initial_state), True
                )
                initial_state[index] = FeaturesDetector.find_lowest_energy(
                    image, initial_state[index], initial_state[(index + 1) % length],
                    initial_state[(index - 1) % length], alpha, beta, gamma,
                    FeaturesDetector.average_distance(initial_state), False
                )

        return initial_state

    @staticmethod
    def find_lowest_energy(image, position, next_point, prev_point, alpha, beta, gamma, avg_distance, first=True):
        width, height = image.shape
        direction = {
            (0, 1): 0,
            (1, 1): 3,
            (1, 0): 2,
            (1, -1): 1,
            (0, -1): 0,
            (-1, -1): 3,
            (-1, 0): 2,
            (-1, 1): 1,
        }
        x, y = position
        current_max = ((0, 0), 0)
        for x_diff in [-1, 0, 1]:
            for y_diff in [-1, 0, 1]:
                current_x = x + x_diff
                current_y = y + y_diff
                if width > current_x >= 0 and height > current_y >= 0:
                    current_value = FeaturesDetector.__get_total_energy(
                        image, (current_x, current_y), next_point, prev_point, alpha, beta,
                        gamma, direction[(x_diff, y_diff)], avg_distance, first)
                    if current_value > current_max[1]:
                        current_max = (x_diff, y_diff), current_value
        return current_max[0]

    @staticmethod
    def __get_total_energy(image, position, next_point, prev_point, alpha, beta, gamma,
                           direction, avg_distance, first=True):
        if first:
            curvature_energy = FeaturesDetector.curvature_energy(position, prev_point, next_point)
        else:
            curvature_energy = FeaturesDetector.second_curvature_energy(position, prev_point, next_point)
        return (alpha * FeaturesDetector.continuity_energy(position, avg_distance, next_point)
                + beta * curvature_energy
                + gamma * FeaturesDetector.image_energy(image, position, direction))

#
# image = Util.load_image('milo.JPG')
# gray = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
#
# image = cv2.drawKeypoints(gray, kp, image)  # , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv2.imwrite('sift_keypoints.jpg', image)
