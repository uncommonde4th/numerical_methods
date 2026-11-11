import numpy as np
import matplotlib.pyplot as plt
from tools import read_numbers, solve_system, print_polynomial


def least_squares(x, y, degree):
    """Нормальная система МНК. Возвращает коэффициенты a0..am и сумму квадратов ошибок Ф"""
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(x ** (i + j))
        B[i] = np.sum(y * x**i)

    coefficients = solve_system(A, B)
    phi = 0.0
    for i in range(len(x)):
        value = 0.0
        for j in range(degree + 1):
            value += coefficients[j] * x[i] ** j
        phi += (y[i] - value) ** 2
    return coefficients, phi


def main():
    data = read_numbers("data_33.txt")
    n = int(data[0])
    x = data[1 : 1 + n]
    y = data[1 + n : 1 + 2 * n]
    precision = int(data[1 + 2 * n])
    degrees = data[2 + 2 * n :].astype(int)

    results = []

    for degree in degrees:
        degree = int(degree)
        coefficients, phi = least_squares(x, y, degree)
        mse = phi / len(x)
        results.append((coefficients, mse))
        print(f"\nМногочлен степени {degree}:")
        print("Коэффициенты:")
        for j in range(len(coefficients)):
            print(f"a{j} = {coefficients[j]:.{precision}f}")
        print_polynomial(coefficients, f"P{degree}", precision)
        print(f"Ф{degree} (сумма квадратов ошибок) = {phi:.{precision}f}")
        print(f"MSE{degree} = {mse:.{precision}f}")

    print("\nСравнение ошибок:")
    for i in range(len(results)):
        print(f"MSE{degrees[i]} = {results[i][1]:.{precision}f}")
    best = min(range(len(results)), key=lambda i: results[i][1])
    print(
        f"Наименьшая ошибка у многочлена степени {degrees[best]} "
        f"(MSE{degrees[best]}), он лучше всех приближает исходные данные."
    )

    xx = np.linspace(x[0], x[-1], 400)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", label="Исходные данные")
    for i in range(len(results)):
        coefficients, mse = results[i]
        yy = np.zeros(len(xx))
        for j in range(len(coefficients)):
            yy += coefficients[j] * xx**j
        plt.plot(xx, yy, label=f"P{degrees[i]}(x), MSE{degrees[i]} = {mse:.5f}")
    plt.title("Метод наименьших квадратов")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig("least_squares.png")
    plt.close()
    print("График сохранён: least_squares.png")


if __name__ == "__main__":
    main()