import numpy as np
import matplotlib.pyplot as plt
from tools import read_numbers, solve_system, print_polynomial


def least_squares(x, y, degree):
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(x ** (i + j))
        B[i] = np.sum(y * x**i)

    coefficients = solve_system(A, B)
    error = 0.0
    for i in range(len(x)):
        value = 0.0
        for j in range(degree + 1):
            value += coefficients[j] * x[i] ** j
        error += (y[i] - value) ** 2
    return coefficients, error / len(x)


def main():
    data = read_numbers("data_33.txt")
    n = 6
    x = data[:n]
    y = data[n : 2 * n]
    degrees = data[2 * n :].astype(int)

    results = []

    for degree in degrees:
        coefficients, mse = least_squares(x, y, int(degree))
        results.append((coefficients, mse))
        print(f"\nМногочлен степени {degree}:")
        print_polynomial(coefficients, f"P{degree}")
        print(f"Средняя квадратичная ошибка = {mse:.12f}")

    best = min(range(len(results)), key=lambda i: results[i][1])
    print(f"\nНаименьшая ошибка у многочлена степени {degrees[best]}.")

    xx = np.linspace(x[0], x[-1], 400)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", label="Исходные данные")
    for i, (coefficients, mse) in enumerate(results):
        yy = np.zeros(len(xx))
        for j in range(len(coefficients)):
            yy += coefficients[j] * xx**j
        plt.plot(xx, yy, label=f"Степень {int(degrees[i])}, MSE={mse:.5f}")
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
