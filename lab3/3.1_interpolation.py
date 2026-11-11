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
        for i in range(len(x)):
            print(f"x[{i}] = {x[i]:.{precision}f}, y[{i}] = {y[i]:.{precision}f}")
        print("\nМногочлен Лагранжа:")
        print_polynomial(l_coef, "L", precision)
        print("Многочлен Ньютона:")
        print_polynomial(n_coef, "P", precision)
        print(f"L(x*) = {l_value:.{precision}f}")
        print(f"P(x*) = {n_value:.{precision}f}")
        print(f"f(x*) = {exact:.{precision}f}")
        print(f"Погрешность Лагранжа = {abs(exact-l_value):.{precision}f}")
        print(f"Погрешность Ньютона = {abs(exact-n_value):.{precision}f}")

        xx = np.linspace(x[0], x[-1], 400)
        plt.figure(figsize=(8, 5))
        plt.plot(xx, f(xx), label="f(x) = tg(x)")
        plt.plot(xx, np.polyval(l_coef[::-1], xx), "--", label="Лагранж")
        plt.plot(xx, np.polyval(n_coef[::-1], xx), ":", label="Ньютон")
        plt.scatter(x, y, color="black", label="Узлы")
        plt.scatter(x_star, l_value, color="red", label="x*")
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
