import math

import matplotlib.pyplot as plt

# import numpy as np
import numpy as np

points = []


def linear():
    sx, sy, sxx, sxy = 0, 0, 0, 0
    n = len(points)
    for p in points:
        sx += p[0]
        sy += p[1]
        sxx += pow(p[0], 2)
        sxy += p[0] * p[1]
    x_mid = sx / len(points)
    y_mid = sy / len(points)
    summ1 = 0
    summ2 = 0
    summ3 = 0
    for p in points:
        summ1 += (p[0] - x_mid) * (p[1] - y_mid)
        summ2 += (p[0] - x_mid) ** 2
        summ3 += (p[1] - y_mid) ** 2
    r = summ1 / math.sqrt(summ2 * summ3)
    print("Коэффициент корреляции = {}".format(r))
    delta = sxx * n - sx * sx
    delta1 = sxy * n - sx * sy
    delta2 = sxx * sy - sx * sxy
    a = delta1 / delta
    b = delta2 / delta
    s = 0
    sigma = 0

    f = lambda t: a * t + b
    for p in points:
        s += (f(p[0]) - p[1]) ** 2

    sigma = math.sqrt(s / n)
    return a, b, s, sigma, f


def square():
    sx, sx2, sx3, sx4, sy, sxy, sx2y = 0, 0, 0, 0, 0, 0, 0
    n = len(points)
    for p in points:
        sx += p[0]
        sx2 += pow(p[0], 2)
        sx3 += pow(p[0], 3)
        sx4 += pow(p[0], 4)
        sy += p[1]
        sxy += p[0] * p[1]
        sx2y += pow(p[0], 2) * p[1]

    m_list = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
    A = np.array(m_list)
    B = np.array([sy, sxy, sx2y])
    a0, a1, a2 = np.linalg.solve(A, B)
    s = 0
    sigma = 0
    f = lambda x: x * x * a2 + x * a1 + a0
    for p in points:
        s += (f(p[0]) - p[1]) ** 2

    sigma = math.sqrt(s / n)
    return a2, a1, a0, s, sigma, f


def deg():
    sx, sy, sxx, sxy = 0, 0, 0, 0
    n = len(points)
    for p in points:
        if p[0] > 0 and p[1] > 0:
            sx += np.log(p[0])
            sy += np.log(p[1])
            sxx += pow(np.log(p[0]), 2)
            sxy += np.log(p[0]) * np.log(p[1])
        else:
            print("Значения должны быть больше 0")
            return
    delta = sxx * n - sx * sx
    delta1 = sxy * n - sx * sy
    delta2 = sxx * sy - sx * sxy
    b = delta1 / delta

    a = delta2 / delta
    a = np.exp(a)
    s = 0
    sigma = 0
    f = lambda x: a * pow(x, b)
    for p in points:
        s += (f(p[0]) - p[1]) ** 2

    sigma = math.sqrt(s / n)
    return a, b, s, sigma, f


def exp():
    sx, sy, sxx, sxy = 0, 0, 0, 0
    n = len(points)
    for p in points:
        sx += p[0]
        sy += np.log(p[1])
        sxx += pow(p[0], 2)
        sxy += p[0] * np.log(p[1])
    delta = sxx * n - sx * sx
    delta1 = sxy * n - sx * sy
    delta2 = sxx * sy - sx * sxy
    b = delta1 / delta
    a = delta2 / delta
    a = np.exp(a)
    s = 0
    sigma = 0
    f = lambda x: a * np.e ** (x * b)
    for p in points:
        s += (f(p[0]) - p[1])**2

    sigma = np.sqrt(s/n)

    return a, b, s, sigma, f


def log():
    sx, sy, sxx, sxy = 0, 0, 0, 0
    n = len(points)
    for p in points:
        sx += np.log(p[0])
        sy += (p[1])
        sxx += np.log(p[0]) ** 2
        sxy += np.log(p[0]) * (p[1])
    delta = sxx * n - sx * sx
    delta1 = sxy * n - sx * sy
    delta2 = sxx * sy - sx * sxy
    a = delta1 / delta
    b = delta2 / delta
    s = 0
    f = lambda x: a * np.log(x) + b
    sigma = 0
    for p in points:
        s += (f(p[0]) - p[1])**2

    sigma = math.sqrt(s/n)
    return a, b, s, sigma, f


def print_graph(func, d1, d2):
    fig, ax = plt.subplots()
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x2 = np.linspace(min(x) - d1, max(x) + d2, 1000)
    y2 = func(x2)
    ax.plot(x2, y2)
    plt.scatter(x, y)
    plt.show()


def print_all(func1, func2, func3, func4, func5):
    fig, ax = plt.subplots()
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x1 = np.linspace(min(x), max(x), 1000)

    ax.plot(x1, func1(x1), color='b', label = "линейная")
    ax.plot(x1, func2(x1), color='g', label = "степенная")
    ax.plot(x1, func3(x1), color='r', label = "экспоненциальная")
    ax.plot(x1, func4(x1), color='y', label = "логорифмическая")
    ax.plot(x1, func5(x1), color='m', label = "квадратичная")
    ax.legend()
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    print("Выберите источник ввода:")
    print("1 - Консоль")
    print("2 - Файл")
    inp = input()

    if inp == "1":
        while True:
            n = int(input("Введите количество вводимых строк (минимум 12): "))
            if n >= 12:
                for i in range(n):
                    x, y = map(float, input().split(' '))
                    points += [[x, y]]
                break
    elif inp == "2":
        with open("square.txt", 'r') as file:
            for line in file:
                x, y = map(float, line.split(' '))
                points += [[x, y]]

    a, b, s, sigma1, f1 = linear()
    bestF, bestS = f1, sigma1
    mess = "Линейная функция"
    print("+" + "-"*23 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+")
    print("|    Вид функции        |              a              |              b              |              c              |              S              |             СКО             |")
    print("+" + "-"*23 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+")
    print(
        "| \u03c6(x) = a * x + b      |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |".format(a, b, 0.,
                                                                                                               s,
                                                                                                               sigma1))

    a, b, s, sigma2, f2 = deg()
    if sigma2 < bestS:
        bestF = f2
        bestS = sigma2
        mess = "степенная функция"
    print("| \u03c6(x) = a * x\u1d47         |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |".format(a,
                                                                                                                      b,
                                                                                                                      0.,
                                                                                                                      s,
                                                                                                                      sigma2))
    a, b, s, sigma3, f3 = exp()
    if sigma3 < bestS:
        bestS = sigma3
        bestF = f3
        mess = "экспаненциальная функция"
    print(
        "| \u03c6(x) = a * e\u1d47\u1d61        |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |".format(
            a, b, 0., s, sigma3))
    a, b, s, sigma4, f4 = log()
    if sigma4 < bestS:
        bestS = sigma4
        bestF = f4
        mess = "логорифмическая функция"
    print(
        "| \u03c6(x) = a * ln(x) + b  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |".format(a, b, 0.,
                                                                                                               s,
                                                                                                               sigma4))
    a, b, c, s, sigma5, f5 = square()

    if sigma5 < bestS:
        bestS = sigma5
        bestF = f5
        mess = "многочлен второй степени"
    print("| \u03c6(x) = ax\u00b2 + bx + c   |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |  {: ^25.15f}  |".format(a,
                                                                                                                      b,
                                                                                                                      c,
                                                                                                                      s,
                                                                                                                      sigma5))
    print("+" + "-"*23 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+" + "-"*29 + "+")
    print_all(f1, f2, f3, f4, f5)
    print(mess)


    print_graph(bestF, 0, 0)
