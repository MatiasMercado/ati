import cv2


class Util:
    PATH = '../../resources/lena.ascii.pbm';

    @staticmethod
    def load_gray_scale(path):
        return cv2.imread(path)[:, :, 1]

    @staticmethod
    def trim(image, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return image[min(x1, x2):max(x1, x2), min(y1, y2):max(y1, y2)]

    @staticmethod
    def get_info(image, p1, p2):
        aux = Util.trim(image, p1, p2)
        count = len(aux)
        return float(aux.sum()) / max(count, 1), count

    @staticmethod
    def save(image, name):
        cv2.imwrite(name + ".pbm", image, (cv2.IMWRITE_PXM_BINARY, 0))

PATH = '../../resources/lena.ascii.pbm'
img = Util.load_gray_scale(PATH)
print(img.shape)
img = Util.trim(img, (30, 40), (60, 85))
print(img.shape)
info = Util.get_info(img, (3, 3), (7, 7))
print(info)
Util.save(img, "aver")
