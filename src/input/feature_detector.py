from cv2 import cv2


class FeaturesDetector:
    @staticmethod
    def SIFT(img):
        gray = cv2.cvtColor(img.astype('B'), cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)

        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img

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
