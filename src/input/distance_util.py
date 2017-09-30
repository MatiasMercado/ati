class DistanceUtil:
    @staticmethod
    def sqr_euclidean_distance(p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    @staticmethod
    def euclidean_distance_lower_than(p1, p2, threshold):
        if DistanceUtil.sqr_euclidean_distance(p1, p2) < threshold ** 2:
            return True
        return False

    @staticmethod
    def chebyshev_distance(p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return max(abs(x1 - x2), abs(y1 - y2))

    @staticmethod
    def chebyshev_distance_lower_than(p1, p2, threshold):
        if DistanceUtil.chebyshev_distance(p1, p2) < threshold:
            return True
        return False
