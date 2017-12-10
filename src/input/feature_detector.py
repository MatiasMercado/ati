import numpy as np
from cv2 import cv2


class FeaturesDetector:
    @staticmethod
    def SIFT(img):
        gray = cv2.cvtColor(img.astype('B'), cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)

        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img

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
