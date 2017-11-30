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
