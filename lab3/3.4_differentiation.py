import numpy as np
from tools import read_numbers, divided_differences


def newton_derivatives(x, y, value):
    table = divided_differences(x, y)
    coefficients = table[0, :]
    p = coefficients[-1]
    dp = 0.0
    d2p = 0.0

    for i in range(len(x) - 2, -1, -1):
        d2p = d2p * (value - x[i]) + 2 * dp
        dp = dp * (value - x[i]) + p
        p = p * (value - x[i]) + coefficients[i]

    return dp, d2p


def main():
    data = read_numbers("data_34.txt")
    n = 5
    x = data[:n]
    y = data[n : 2 * n]
    x_star = data[2 * n]
    precision = int(data[2 * n + 1])

    print("\nИсходная таблица:")
    for i in range(n):
        print(f"x[{i}] = {x[i]:.{precision}f}, y[{i}] = {y[i]:.{precision}f}")

    first, second = newton_derivatives(x, y, x_star)
    print(f"\nx* = {x_star:.{precision}f}")
    print(f"Первая производная: f'(x*) = {first:.{precision}f}")
    print(f"Вторая производная: f''(x*) = {second:.{precision}f}")


if __name__ == "__main__":
    main()
