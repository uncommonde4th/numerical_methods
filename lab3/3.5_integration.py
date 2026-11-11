import numpy as np
from tools import read_numbers


def f(x):
    return x / (3 * x + 4) ** 3


def count_steps(a, b, h):
    n = int(round((b - a) / h))
    if abs(n * h - (b - a)) > 1e-9:
        raise ValueError("Шаг h должен укладываться в отрезок целое число раз")
    return n


def left_rectangles(a, b, h):
    n = count_steps(a, b, h)
    result = 0.0
    for i in range(n):
        result += f(a + i * h)
    return result * h


def right_rectangles(a, b, h):
    n = count_steps(a, b, h)
    result = 0.0
    for i in range(1, n + 1):
        result += f(a + i * h)
    return result * h


def middle_rectangles(a, b, h):
    n = count_steps(a, b, h)
    result = 0.0
    for i in range(n):
        result += f(a + (i + 0.5) * h)
    return result * h


def trapezoids(a, b, h):
    n = count_steps(a, b, h)
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def simpson(a, b, h):
    n = count_steps(a, b, h)
    if n % 2 != 0:
        raise ValueError("Для метода Симпсона число интервалов должно быть чётным")
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
    k = h1 / h2

    methods = [
        ("Левые прямоугольники", left_rectangles, 1),
        ("Правые прямоугольники", right_rectangles, 1),
        ("Средние прямоугольники", middle_rectangles, 2),
        ("Трапеции", trapezoids, 2),
        ("Симпсон", simpson, 4),
    ]

    exact = exact_integral(a, b)
    print(f"Интервал: [{a:g}, {b:g}], h1 = {h1:g}, h2 = {h2:g}")
    print(f"Точное значение интеграла: {exact:.{precision}f}")

    width = precision + 6
    print("\nРезультаты численного интегрирования:")
    print(
        f"{'Метод':<24}{'F(h1)':>{width}}{'F(h2)':>{width}}{'p':>4}"
        f"{'Рунге-Ромберг':>{width}}{'Оценка погр. (h2)':>{width}}"
    )

    results = []
    for name, method, order in methods:
        value1 = method(a, b, h1)
        value2 = method(a, b, h2)
        refined = value2 + (value2 - value1) / (k**order - 1)
        estimate = abs(value2 - value1) / (k**order - 1)
        results.append((name, value1, value2, refined))
        print(
            f"{name:<24}{value1:>{width}.{precision}f}{value2:>{width}.{precision}f}"
            f"{order:>4}{refined:>{width}.{precision}f}{estimate:>{width}.3e}"
        )

    print("\nАбсолютная погрешность относительно точного значения:")
    print(
        f"{'Метод':<24}{'при h1':>{width}}{'при h2':>{width}}{'Рунге-Ромберг':>{width}}"
    )
    for name, value1, value2, refined in results:
        print(
            f"{name:<24}{abs(exact - value1):>{width}.3e}"
            f"{abs(exact - value2):>{width}.3e}{abs(exact - refined):>{width}.3e}"
        )

    best_h2 = min(results, key=lambda item: abs(exact - item[2]))
    best_refined = min(results, key=lambda item: abs(exact - item[3]))
    print(f"\nСамый точный метод при шаге h2: {best_h2[0]}")
    print(f"Самый точный метод после уточнения Рунге-Ромберга: {best_refined[0]}")


if __name__ == "__main__":
    main()