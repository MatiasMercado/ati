import numpy as np
from cv2 import cv2

from src.input.filter_provider import FilterProvider
from src.input.logGabor import LogGabor
from src.input.util import Util
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
    def circleness_energy(control_point, prev_point, next_point, center):
        cp_distane = VectorUtil.sqr_euclidean_distance(center, control_point)
        pp_distance = VectorUtil.sqr_euclidean_distance(center, prev_point)
        np_distance = VectorUtil.sqr_euclidean_distance(center, next_point)
        return np.abs(cp_distane * 2 - pp_distance - np_distance)

    @staticmethod
    def continuity_energy(control_point, average_distance, prev_point):
        return np.abs(average_distance - np.sqrt(VectorUtil.sqr_euclidean_distance(control_point, prev_point)))

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

        if hi_abs != 0:
            hi_normilzed = (hi[0] / hi_abs, hi[1] / hi_abs)
        else:
            hi_normilzed = hi

        if hi1_abs != 0:
            hi1_normilzed = (hi1[0] / hi1_abs, hi1[1] / hi1_abs)
        else:
            hi1_normilzed = hi1

        return VectorUtil.vector_abs((hi1_normilzed[0] - hi_normilzed[0],
                                      hi1_normilzed[1] - hi_normilzed[1]))

    @staticmethod
    def image_energy(image, position, direction):
        return np.max([
            FilterProvider.single_point_gradient(image, position, 0, weighted=True),
            FilterProvider.single_point_gradient(image, position, 1, weighted=True),
            FilterProvider.single_point_gradient(image, position, 2, weighted=True),
            FilterProvider.single_point_gradient(image, position, 3, weighted=True)
        ])

    @staticmethod
    def iris_detector(image, initial_state_iris, initial_state_pupil, alpha=0.75, beta=0.75, gamma=0.9, iterations=1):
        iris_length = len(initial_state_iris)
        pupil_length = len(initial_state_pupil)
        original = image
        image = Util.load_image('./input/myCircles/ojo-anisotropic-60-0-3.jpg')
        image = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)

        # image_editor = ImageEditor()
        print("Detecting pupil")
        for i in range(iterations):
            for index in range(pupil_length):
                initial_state_pupil[index] = FeaturesDetector.find_lowest_energy(
                    image, initial_state_pupil[index], initial_state_pupil[(index + 1) % pupil_length],
                    initial_state_pupil[(index - 1) % pupil_length], alpha, beta, gamma,
                    FeaturesDetector.average_distance(initial_state_pupil), True
                )

                # if np.abs(last[0] - initial_state[index][0]) > 1 or np.abs(last[1] - initial_state[index][1]) > 1:
                #     print(last)
                #     print(initial_state[index])

                # initial_state[index] = FeaturesDetector.find_lowest_energy(
                #     image, initial_state[index], initial_state[(index + 1) % length],
                #     initial_state[(index - 1) % length], alpha, beta, gamma,
                #     FeaturesDetector.average_distance(initial_state), False
                # )
            if i % 10 == 0:
                # transformed_image = image_editor.draw_control_points(image, initial_state)
                # cv2.imwrite('./myCircles/myCircle-' + str(i) + '.jpg', transformed_image)
                beta += 0.01
                print(i)

        print("Detecting iris")
        for i in range(iterations):
            for index in range(iris_length):
                initial_state_iris[index] = FeaturesDetector.find_lowest_energy(
                    image, initial_state_iris[index], initial_state_iris[(index + 1) % iris_length],
                    initial_state_iris[(index - 1) % iris_length], alpha, beta, gamma,
                    FeaturesDetector.average_distance(initial_state_iris), True
                )

                # if np.abs(last[0] - initial_state[index][0]) > 1 or np.abs(last[1] - initial_state[index][1]) > 1:
                #     print(last)
                #     print(initial_state[index])

                # initial_state[index] = FeaturesDetector.find_lowest_energy(
                #     image, initial_state[index], initial_state[(index + 1) % length],
                #     initial_state[(index - 1) % length], alpha, beta, gamma,
                #     FeaturesDetector.average_distance(initial_state), False
                # )
            if i % 10 == 0:
                # transformed_image = image_editor.draw_control_points(image, initial_state)
                # cv2.imwrite('./myCircles/myCircle-' + str(i) + '.jpg', transformed_image)
                beta += 0.01
                print(i)
        # print('finished')
        iris = LogGabor.normalization(original, initial_state_pupil, initial_state_iris)
        print('iris')
        print(iris)
        cv2.imwrite('./input/result/iris.jpg', iris)
        snipped_iris = LogGabor.interest_degrees(iris)
        print('snipped_iris')
        cv2.imwrite('./input/result/snipped.jpg', snipped_iris)
        print(snipped_iris)
        filters = LogGabor.build_filters()
        template = LogGabor.process(snipped_iris, filters)
        print('template')
        print(template)
        cv2.imwrite('./input/result/template.jpg', template)

        return initial_state_iris, initial_state_pupil

    @staticmethod
    def find_lowest_energy(image, position, next_point, prev_point, alpha_v, beta_v, gamma_v, avg_distance, first=True):
        width, height = image.shape
        # direction = {
        #     (0, 1): 0,
        #     (1, 1): 3,
        #     (1, 0): 2,
        #     (1, -1): 1,
        #     (0, -1): 0,
        #     (-1, -1): 3,
        #     (-1, 0): 2,
        #     (-1, 1): 1,
        # }
        direction = {
            (-1, 0): 0,
            (-1, 1): 1,
            (0, 1): 2,
            (1, 1): 3,
            (1, 0): 0,
            (1, -1): 1,
            (0, -1): 2,
            (-1, -1): 3,
        }
        x, y = position
        energies = []
        max_continuity = 0
        max_curvature = 0
        max_gradient = 0
        min_gradient = 0
        for x_diff in [-1, 0, 1]:
            for y_diff in [-1, 0, 1]:
                current_x = x + x_diff
                current_y = y + y_diff
                if width > current_x >= 0 and height > current_y >= 0 and not (x_diff == 0 and y_diff == 0):
                    current_energies = FeaturesDetector.__get_total_energy(
                        image, (current_x, current_y), next_point, prev_point, alpha_v, beta_v,
                        gamma_v, direction[(x_diff, y_diff)], avg_distance, first)

                    energies.append(((current_x, current_y), current_energies))
                    max_continuity = np.max([current_energies[0], max_continuity])
                    max_curvature = np.max([current_energies[1], max_curvature])
                    max_gradient = np.max([current_energies[2], max_gradient])
                    min_gradient = np.min([current_energies[2], min_gradient])

        lowest_energy = ((0, 0), 100000)
        for energy in energies:
            if (max_gradient - min_gradient) != 0:
                current_normalized_gradient = (energy[1][2] - min_gradient) / (max_gradient - min_gradient)
            else:
                current_normalized_gradient = 0
            # current_normalized_gradient = ((energy[1][2] - min_gradient) / (max_gradient - min_gradient)) if (
            #     max_gradient - min_gradient) != 0 else 0

            current_energy = (energy[1][0] / max_continuity * alpha_v + energy[1][1] / max_curvature * beta_v
                              - current_normalized_gradient * gamma_v)
            if current_energy < lowest_energy[1]:
                lowest_energy = (energy[0], current_energy)

        return lowest_energy[0]

    @staticmethod
    def __get_total_energy(image, position, next_point, prev_point, alpha, beta, gamma,
                           direction, avg_distance, first=True):
        if first:
            curvature_energy = FeaturesDetector.curvature_energy(position, prev_point, next_point)
        else:
            curvature_energy = FeaturesDetector.second_curvature_energy(position, prev_point, next_point)
        continuity_energy = FeaturesDetector.continuity_energy(position, avg_distance, prev_point)
        image_energy = FeaturesDetector.image_energy(image, position, direction)

        return continuity_energy, curvature_energy, image_energy

# def process_input(params):
#     image_editor = ImageEditor()
#     image = Util.load_image('./myCircles/ojo-anisotropic-60-0-3.jpg')
#
#     gray = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
#
#     alpha_param, beta_param, gamma_param = params
#     initial_state = Provider.get_circle_coordinates(70, (124, 175))
#     control_points = FeaturesDetector.iris_detector(gray, initial_state, iterations=1,
#                                                     alpha=alpha_param, beta=beta_param, gamma=gamma_param)
#     transformed_image = image_editor.draw_control_points(gray, control_points)
#     cv2.imwrite('./myCircles/result-' + str(alpha_param) + '-' + str(beta_param) + '-' + str(gamma_param) + '.jpg',
#                 transformed_image)
#
#
# num_cores = multiprocessing.cpu_count()
#
# inputs = []
# for a in np.arange(0.3, 0.9, 0.2):
#     for b in np.arange(0.7, 1.1, 0.05):
#         for g in np.arange(0.7, 1.1, 0.1):
#             inputs.append((a, b, g))
#
#
# class myThread(threading.Thread):
#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#
#     def run(self):
#         print("Starting " + str(self.params))
#         image = Util.load_image('./myCircles/ojo-anisotropic-60-0-3.jpg')
#
#         gray = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
#
#         alpha_param, beta_param, gamma_param = self.params
#         initial_state = Provider.get_circle_coordinates(70, (124, 175))
#         control_points = FeaturesDetector.iris_detector(gray, initial_state, iterations=80,
#                                                         alpha=alpha_param, beta=beta_param, gamma=gamma_param)
#         transformed_image = image_editor.draw_control_points(gray, control_points)
#         cv2.imwrite('./myCircles/result-' + str(alpha_param) + '-' + str(beta_param) + '-' + str(gamma_param) + '.jpg',
#                     transformed_image)
#         print("Exiting " + str(self.params))
#
#
# # threads = []
# image_editor = ImageEditor()
#
# # for input_param in inputs:
# #     t = myThread(input_param)
# #     t.start()
# #     threads.append(t)
# #
# # count = 0
# # for t in threads:
# #     count += 1
# #     t.join()
# #     print(str(count))
#
# # print('Finished')
# # image_editor = ImageEditor()
# # image = Util.load_image('ojo.bmp')
# image = Util.load_image('./myCircles/ojo-anisotropic-60-0-3.jpg')
# # # image = Provider.draw_circle()
# # image = FilterProvider.median_filter(image, independent_layer=True)
# # image = FilterProvider.anisotropic_filter(image, 60, 0, 3, independent_layer=True)
# # cv2.imwrite('./myCircles/ojo-anisotropic-60-0-3.jpg', image)
# gray = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
#
# # results = Parallel(n_jobs=num_cores)(delayed(process_input)(i) for i in inputs)
# #
# # initial_state = Provider.get_circle_coordinates(63, (124, 174))
# initial_state = Provider.get_circle_coordinates(70, (124, 175))
# # # print(initial_state)
# control_points = FeaturesDetector.iris_detector(gray, initial_state, iterations=140, alpha=0.7, beta=0.75, gamma=0.9)
# # control_points = initial_state
# transformed_image = image_editor.draw_control_points(gray, control_points)
# # image_editor.create_new_image(transformed_image)
# # print(control_points)
# cv2.imwrite('./myCircles/myCircle-0.7-0.75-0.9.jpg', transformed_image)
