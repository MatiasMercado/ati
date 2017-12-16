import numpy as np
from cv2 import cv2

from src.input.feature_detector import FeaturesDetector
from src.input.provider import Provider
from src.input.util import Util
from src.input.vector_util import VectorUtil

rand = np.random


def evaluate(current, original):
    acu = 0
    for original_index in range(len(original)):
        current_min = 1000000
        for point in current:
            current_min = np.min(
                [current_min, VectorUtil.sqr_euclidean_distance(original[original_index], point)])
        acu += current_min
    return acu


def mutate(methods):
    r = rand.randint(0, 11)
    index = r % 6
    if index >= 2:
        change = 0.1
    elif index == 1:
        change = 1
    else:
        change = 5

    if methods[index] == 0 or r <= 5:
        methods[index] += change
    else:
        if index == 1:
            methods[index] = 0
        else:
            methods[index] -= change


def mutate_simple(methods):
    r = rand.randint(0, 5)
    index = r % 3
    if r <= 2 or methods[index] < 0.1:
        methods[index] += 0.1
    else:
        methods[index] -= 0.1


def cross(method1, method2):
    new_method = []
    r = rand.random()
    if r < 0.5:
        new_method.append(method1.methods[0])
    else:
        new_method.append(method2.methods[0])
    r = rand.random()
    if r < 0.5:
        new_method.append(method1.methods[1])
    else:
        new_method.append(method2.methods[1])
    r = rand.random()
    if r < 0.5:
        new_method.append(method1.methods[2])
    else:
        new_method.append(method2.methods[2])
    r = rand.random()
    if r < 0.5:
        mutate_simple(new_method)
    return Method(new_method)


def select(input_pool=[]):
    input_pool.sort(key=lambda m: m.fitness)
    print(input_pool[0].methods, input_pool[0].fitness)
    with open("src/input/results.txt", "a") as file:
        file.write(str(input_pool[0].methods) + " " + str(input_pool[0].fitness) + "\n")
    new_pool = input_pool[:5]
    children = []
    rand.shuffle(input_pool)
    for j in range(10):
        children.append(cross(input_pool[j * 2], input_pool[j * 2 + 1]))
    new_pool.append(children)
    new_pool.append(input_pool[15:])
    return new_pool


class Method:

    def __init__(self, methods=[]):
        self.fitness = 0
        if len(methods) != 0:
            self.methods = methods  # anisotropic, median, gaussian blur, alpha, beta, gamma
        else:
            self.methods = Method.__random_method()

    def get_methods(self):
        return list(self.methods)

    @staticmethod
    def __random_method():
        r = rand
        return [r.random(), r.random(), r.random()]
        # return [r.randint(0, 8) * 5, r.randint(0,1), r.random() * 0.5, r.random, r.random, r.random]

    def set_fitness(self, fitness):
        self.fitness = fitness

    def apply(self, input_image, input_circle):
        return FeaturesDetector.iris_detector(input_image, input_circle, self.methods[0], self.methods[1],
                                              self.methods[2], 160)


pool = []
for i in range(20):
    pool.append(Method([]))

image = Util.load_image('src/input/myCircles/ojo-anisotropic-60-0-3.jpg')
initial_state = Provider.get_circle_coordinates(70, (124, 175))
ideal = Provider.get_circle_coordinates(63, (124, 174))
gray = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)

with open("src/input/results.txt", "a") as myfile:
    myfile.write("Results:\n")

for k in range(20):
    print("iteration: " + str(k))
    f = 1
    for method in pool:
        print("Method " + str(f))
        f += 1
        method.set_fitness(evaluate(method.apply(gray, list(initial_state)), initial_state))

    pool = select(pool)
