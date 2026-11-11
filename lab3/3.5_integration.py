import numpy as np
from tools import read_numbers


def f(x):
    return x / (3 * x + 4) ** 3


def left_rectangles(a, b, h):
    n = int(round((b - a) / h))
    result = 0.0
    for i in range(n):
        result += f(a + i * h)
    return result * h


def right_rectangles(a, b, h):
    n = int(round((b - a) / h))
    result = 0.0
    for i in range(1, n + 1):
        result += f(a + i * h)
    return result * h


def middle_rectangles(a, b, h):
    n = int(round((b - a) / h))
    result = 0.0
    for i in range(n):
        result += f(a + (i + 0.5) * h)
    return result * h


def trapezoids(a, b, h):
    n = int(round((b - a) / h))
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def simpson(a, b, h):
    n = int(round((b - a) / h))
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 4 * f(a + i * h)
    return result * h / 3


def exact_integral(a, b):
    def primitive(x):
        u = 3 * x + 4
        return (-1 / u + 2 / u**2) / 9

    return primitive(b) - primitive(a)


def main():
    data = read_numbers("data_35.txt")
    a, b, h1, h2 = data[:4]
    precision = int(data[4])

    methods = [
        ("Левые прямоугольники", left_rectangles, 1),
        ("Правые прямоугольники", right_rectangles, 1),
        ("Средние прямоугольники", middle_rectangles, 2),
        ("Трапеции", trapezoids, 2),
        ("Симпсон", simpson, 4),
    ]

    exact = exact_integral(a, b)
    print(f"Интервал: [{a}, {b}], h1 = {h1}, h2 = {h2}")
    print(f"Точное значение: {exact:.{precision}f}\n")

    results = []
    print(
        "Метод                         F(h1)             F(h2)             "
        "Рунге-Ромберг       Оценка ошибки"
    )
    for name, method, order in methods:
        value1 = method(a, b, h1)
        value2 = method(a, b, h2)
        refined = value2 + (value2 - value1) / (2**order - 1)
        error = abs(value2 - value1) / (2**order - 1)
        exact_error = abs(exact - value2)
        results.append((name, exact_error))
        print(
            f"{name:<28} {value1: .{precision}f}  {value2: .{precision}f}  "
            f"{refined: .{precision}f}  {error:.3e}"
        )
        print(f"  Абсолютная погрешность при h2: {exact_error:.3e}")

    best = min(results, key=lambda item: item[1])
    print(f"\nСамый точный метод при h2: {best[0]}")


if __name__ == "__main__":
    main()
