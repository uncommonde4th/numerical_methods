import numpy as np
import matplotlib.pyplot as plt
from tools import read_numbers, solve_system


def cubic_spline(x, y):
    """Естественный кубический сплайн: S''(x0) = S''(xn) = 0"""
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


def piece_value(x, a, b, c, d, i, point):
    """Значение i-го участка сплайна (нумерация с нуля) в точке point"""
    dx = point - x[i]
    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3


def find_interval(x, value):
    for i in range(len(x) - 1):
        if x[i] <= value <= x[i + 1]:
            return i
    raise ValueError("Точка x* лежит вне отрезка интерполяции")


def main():
    data = read_numbers("data_32.txt")
    n = int(data[0])
    x = data[1 : 1 + n]
    y = data[1 + n : 1 + 2 * n]
    x_star = data[1 + 2 * n]
    precision = int(data[2 + 2 * n])

    a, b, c, d = cubic_spline(x, y)

    print("\nИсходные узлы:")
    for i in range(n):
        print(f"x[{i}] = {x[i]:.{precision}f}, y[{i}] = {y[i]:.{precision}f}")

    print("\nКоэффициенты сплайна")
    print(
        "S_i(x) = a_i + b_i*(x - x_(i-1)) + c_i*(x - x_(i-1))^2 + d_i*(x - x_(i-1))^3,"
    )
    print("x_(i-1) <= x <= x_i")
    width = precision + 6
    print(
        f"{'i':<4}{'интервал':<14}{'a[i]':>{width}}{'b[i]':>{width}}"
        f"{'c[i]':>{width}}{'d[i]':>{width}}"
    )
    for i in range(len(a)):
        interval = f"[{x[i]:g}, {x[i + 1]:g}]"
        print(
            f"{i + 1:<4}{interval:<14}{a[i]:>{width}.{precision}f}"
            f"{b[i]:>{width}.{precision}f}{c[i]:>{width}.{precision}f}"
            f"{d[i]:>{width}.{precision}f}"
        )

    interval = find_interval(x, x_star)
    value = piece_value(x, a, b, c, d, interval, x_star)
    print(f"\nx* = {x_star:.{precision}f}")
    print(
        f"Номер интервала, содержащего x*: {interval + 1} ([{x[interval]:g}, {x[interval + 1]:g}])"
    )
    print(f"S(x*) = {value:.{precision}f}")

    print("\nПроверка прохождения сплайна через узлы:")
    max_error = 0.0
    for k in range(n):
        if k == 0:
            s = piece_value(x, a, b, c, d, 0, x[0])
        else:
            s = piece_value(x, a, b, c, d, k - 1, x[k])
        error = abs(s - y[k])
        max_error = max(max_error, error)
        print(
            f"x[{k}] = {x[k]:.{precision}f}: S(x[{k}]) = {s:.{precision}f}, y[{k}] = {y[k]:.{precision}f}, |S - y| = {error:.3e}"
        )
    if max_error < 1e-9:
        print("Сплайн проходит через все узлы")
    else:
        print("Сплайн НЕ проходит через все узлы")

    last_step = x[-1] - x[-2]
    print("\nПроверка граничных условий (вторая производная на концах):")
    print(f"S''(x0) = {2 * c[0]:.{precision}f}")
    print(f"S''(xn) = {2 * c[-2] + 6 * d[-1] * last_step:.{precision}f}")

    plt.figure(figsize=(8, 5))
    for i in range(len(a)):
        xx = np.linspace(x[i], x[i + 1], 100)
        yy = piece_value(x, a, b, c, d, i, xx)
        plt.plot(xx, yy, label=f"Участок {i + 1}")
    plt.scatter(x, y, color="black", label="Узлы")
    plt.scatter(
        x_star,
        value,
        color="red",
        zorder=5,
        label=f"(x*, S(x*)) = ({x_star:g}, {value:.4f})",
    )
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
