import numpy as np
import matplotlib.pyplot as plt
from tools import (
    read_numbers,
    print_polynomial,
    divided_differences,
    polynomial_product,
)


def f(x):
    return np.tan(x)


def lagrange(x, y, value):
    result = 0.0
    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if i != j:
                term *= (value - x[j]) / (x[i] - x[j])
        result += term
    return result


def newton(x, table, value):
    result = table[0, 0]
    product = 1.0
    for i in range(1, len(x)):
        product *= value - x[i - 1]
        result += table[0, i] * product
    return result


def lagrange_coefficients(x, y):
    result = np.zeros(len(x))
    for i in range(len(x)):
        numerator = np.array([1.0])
        denominator = 1.0
        for j in range(len(x)):
            if i != j:
                numerator = polynomial_product(numerator, np.array([-x[j], 1.0]))
                denominator *= x[i] - x[j]
        result[: len(numerator)] += y[i] * numerator / denominator
    return result


def newton_coefficients(x, table):
    result = np.array([table[0, len(x) - 1]])
    for i in range(len(x) - 2, -1, -1):
        result = polynomial_product(result, np.array([-x[i], 1.0]))
        result[0] += table[0, i]
    return result


def print_divided_differences(x, table, precision):
    """Таблица разделённых разностей: строка i - узел x[i] и разности f(x[i], ..., x[i+k])"""
    n = len(x)
    width = precision + 6
    print("\nТаблица разделённых разностей:")
    line = f"{'i':<3}{'x[i]':>{width}}"
    for k in range(n):
        line += f"{'порядок ' + str(k):>{width}}"
    print(line)
    for i in range(n):
        line = f"{i:<3}{x[i]:>{width}.{precision}f}"
        for k in range(n - i):
            line += f"{table[i, k]:>{width}.{precision}f}"
        print(line)


def print_newton_form(x, table, precision):
    """Многочлен Ньютона через разделённые разности: P(x) = f0 + f01*(x-x0) + ..."""
    result = f"P(x) = {table[0, 0]:.{precision}f}"
    factors = ""
    for i in range(1, len(x)):
        if x[i - 1] >= 0:
            factors += f"(x - {x[i - 1]:.{precision}f})"
        else:
            factors += f"(x + {-x[i - 1]:.{precision}f})"
        sign = "-" if table[0, i] < 0 else "+"
        result += f" {sign} {abs(table[0, i]):.{precision}f}*{factors}"
    print(result)


def main():
    data = read_numbers("data_31.txt")
    x_star = data[0] / data[1] * np.pi
    precision = int(data[2])
    pos = 3

    print(f"x* = {x_star:.{precision}f}")

    for case_name, plot_file in [
        ("а", "interpolation_a.png"),
        ("б", "interpolation_b.png"),
    ]:
        count = int(data[pos])
        pos += 1
        fractions = data[pos : pos + 2 * count].reshape(count, 2)
        pos += 2 * count
        x = fractions[:, 0] / fractions[:, 1] * np.pi
        y = f(x)

        table = divided_differences(x, y)
        l_coef = lagrange_coefficients(x, y)
        n_coef = newton_coefficients(x, table)
        l_value = lagrange(x, y, x_star)
        n_value = newton(x, table, x_star)
        exact = f(x_star)

        print(f"\nСлучай {case_name}")
        print("Узлы интерполяции:")
        for i in range(len(x)):
            print(f"x[{i}] = {x[i]:.{precision}f}, y[{i}] = {y[i]:.{precision}f}")

        print("\nМногочлен Лагранжа:")
        print_polynomial(l_coef, "L", precision)

        print_divided_differences(x, table, precision)

        print("\nМногочлен Ньютона (через разделённые разности):")
        print_newton_form(x, table, precision)
        print("Многочлен Ньютона (в виде суммы степеней x):")
        print_polynomial(n_coef, "P", precision)

        print(f"\nL(x*) = {l_value:.{precision}f}")
        print(f"P(x*) = {n_value:.{precision}f}")
        print(f"f(x*) = {exact:.{precision}f}")
        print(f"Абсолютная погрешность Лагранжа = {abs(exact - l_value):.{precision}f}")
        print(f"Абсолютная погрешность Ньютона = {abs(exact - n_value):.{precision}f}")
        print(f"|L(x*) - P(x*)| = {abs(l_value - n_value):.{precision}e}")

        xx = np.linspace(x[0], x[-1], 400)
        plt.figure(figsize=(8, 5))
        plt.plot(xx, f(xx), label="f(x) = tg(x)")
        plt.plot(xx, np.polyval(l_coef[::-1], xx), "--", label="Лагранж")
        plt.plot(xx, np.polyval(n_coef[::-1], xx), ":", linewidth=2, label="Ньютон")
        plt.scatter(x, y, color="black", label="Узлы")
        plt.axvline(x_star, color="red", linestyle="--", linewidth=0.8)
        plt.scatter(x_star, l_value, color="red", zorder=5, label="x*")
        plt.title(f"Интерполяция, случай {case_name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_file)
        plt.close()
        print(f"График сохранён: {plot_file}")


if __name__ == "__main__":
    main()