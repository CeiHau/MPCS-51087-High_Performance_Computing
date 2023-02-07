import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


def gdc_for_linear(n_points, alpha, epsilon):
    m = random.uniform(-10, 10)
    b = random.uniform(-10, 10)
    n = len(n_points)
    iter = 0
    while True:
        iter += 1
        delta_m = 0
        delta_b = 0
        for point in n_points:
            delta_m += 2 * b * point[0] + 2 * m * point[0] * point[0] - 2 * point[0] * point[1]
            delta_b += 2 * b + 2 * m * point[0] - 2 * point[1]
        delta_m /= n
        delta_b /= n

        m -= alpha * delta_m
        b -= alpha * delta_b
        norm = math.sqrt(delta_m * delta_m + delta_b * delta_b)

        if norm < epsilon:
            return m, b, iter


def gdc_for_second_order(n_points, alpha, epsilon):
    h = random.uniform(-10, 10)
    m = random.uniform(-10, 10)
    b = random.uniform(-10, 10)
    n = len(n_points)
    i = 0
    while True:
        i += 1
        if i % 1000 ==0:
            print(i, h,m,b)
        delta_h = 0
        delta_m = 0
        delta_b = 0
        for point in n_points:
            delta_h += 2.0 * (point[1] - h * (point[0] ** 2) - m * point[0] - b) * (-point[0] ** 2.0)
            delta_m += 2.0 * (point[1] - h * (point[0] ** 2) - m * point[0] - b) * (-point[0])
            delta_b += 2.0 * (point[1] - h * (point[0] ** 2) - m * point[0] - b) * (-1.0)
        delta_h /= n
        delta_m /= n
        delta_b /= n

        h -= (alpha * delta_h)
        m -= (alpha * delta_m)
        b -= (alpha * delta_b)

        norm = math.sqrt((delta_h ** 2) + (delta_m ** 2) + (delta_b ** 2))
        # print(norm)
        if norm < epsilon:
            return h, m, b, i


if __name__ == '__main__':
    points = pd.read_csv("points.txt", sep="\t")

    plt.scatter(points['x'], points['y'])
    x = np.linspace(-10, 10, 200)

    # m, b, i = gdc_for_linear(points.values, 0.02, 0.00001)
    # print("converge in ", i, " iterations")
    # y = m * x + b
    # plt.plot(x, y)
    #
    h, m, b, i = gdc_for_second_order(points.values, 0.00005, 0.001)
    print(h, m, b, i)
    print("converge in ", i, " iterations")
    y = h * (x ** 2) + m * x + b
    plt.plot(x, y)

    plt.show()
    pass
