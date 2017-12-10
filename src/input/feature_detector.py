import numpy as np
from cv2 import cv2


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
        return (p1 - 2*x + n1)**2 + (p2 -2*y + n2)**2

    @staticmethod
    def continuity_energy(control_point, average_distance, prev_point):
        x, y = control_point
        p1, p2 = prev_point
        return average_distance - np.sqrt((x-p1)**2 + (y-p2)**2)

    @staticmethod
    def average_distance(control_points):
        # Set last point as previous for the first element
        prev = control_points[len(control_points)-1]
        average = 0
        for point in control_points:
            x, y = point
            p1, p2 = prev
            average += np.sqrt((x-p1)**2 + (y-p2)**2)
            prev = point
        return average / len(control_points)

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
