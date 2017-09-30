import numpy as np

from src.input.filter_provider import FilterProvider

DEFAULT_ZERO_DETECTOR_THRESHOLD = 5


class BorderDetector:
    @staticmethod
    def laplacian_detector(image):
        mask = np.matrix([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]])
        aux_image = FilterProvider.sliding_window(image=image, mask=mask)
        aux_image = BorderDetector.__zero_detector(aux_image, threshold=4)
        return aux_image

    @staticmethod
    def __zero_detector(image, independent_layer=False, threshold=DEFAULT_ZERO_DETECTOR_THRESHOLD):
        ans = np.zeros(image.shape)
        (image_width, image_height) = image.shape[0], image.shape[1]

        def single_zero_detector(p1, p2):
            if (p1 < 0 != p2 < 0) and np.abs(p1 - p2) > threshold:
                return 255
            return 0

        for x in range(image_width):
            for y in range(image_height - 1):
                if independent_layer:
                    for z in range(image.shape[2]):
                        ans[x, y, z] = single_zero_detector(image[x, y, z], image[x, y + 1, z])
                else:
                    aux = single_zero_detector(image[x, y, 0], image[x, y + 1, 0])
                    for z in range(image.shape[2]):
                        ans[x, y, z] = aux
        return ans

# img = Util.load_raw('LENA.RAW')
# img = BorderDetector.laplacian_detector(img)
# Util.save_raw(img, 'lena_laplacian_border')
