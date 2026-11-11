import numpy as np
import matplotlib.pyplot as plt
from tools import read_numbers, solve_system


def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1.0
    A[n, n] = 1.0

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c = solve_system(A, B)
    a = y[:-1].copy()
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def spline_value(x, a, b, c, d, value):
    interval = len(a) - 1
    for i in range(len(a)):
        if x[i] <= value <= x[i + 1]:
            interval = i
            break
    dx = value - x[interval]
    result = a[interval] + b[interval] * dx + c[interval] * dx**2 + d[interval] * dx**3
    return result, interval


def main():
    data = read_numbers("data_32.txt")
    n = 5
    x = data[:n]
    y = data[n : 2 * n]
    x_star = data[2 * n]
    precision = int(data[2 * n + 1])

    a, b, c, d = cubic_spline(x, y)

    print("\nКоэффициенты сплайна:")
    print("i       a[i]          b[i]          c[i]          d[i]")
    for i in range(len(a)):
        print(
            f"{i+1:<5} {a[i]: .{precision}f} {b[i]: .{precision}f} "
            f"{c[i]: .{precision}f} {d[i]: .{precision}f}"
        )

    value, interval = spline_value(x, a, b, c, d, x_star)
    print(f"\nx* = {x_star:.{precision}f}")
    print(f"Интервал: [{x[interval]}, {x[interval+1]}]")
    print(f"S(x*) = {value:.{precision}f}")
    print(f"S''(x0) = {2*c[0]:.{precision}f}")
    print(f"S''(xn) = {2*c[-1]:.{precision}f}")

    plt.figure(figsize=(8, 5))
    for i in range(len(a)):
        xx = np.linspace(x[i], x[i + 1], 100)
        dx = xx - x[i]
        yy = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        plt.plot(xx, yy, label=f"Участок {i+1}")
    plt.scatter(x, y, color="black", label="Узлы")
    plt.scatter(x_star, value, color="red", label="x*")
    plt.title("Кубический сплайн")
    plt.xlabel("x")
    plt.ylabel("S(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("spline.png")
    plt.close()
    print("График сохранён: spline.png")


if __name__ == "__main__":
    main()
