import numpy as np


def read_numbers(filename):
    """Считывает числа из txt, пропуская пустые строки и комментарии"""
    with open(filename, encoding="utf-8") as f:
        data = []
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                data.extend(float(value) for value in line.split())
    return np.array(data, dtype=float)


def print_polynomial(coefficients, name, precision=6):
    result = name + "(x) = "
    for i in range(len(coefficients) - 1, -1, -1):
        value = coefficients[i]
        if abs(value) < 1e-12:
            continue
        if result.endswith("= "):
            sign = "-" if value < 0 else ""
        else:
            sign = " - " if value < 0 else " + "
        value = abs(value)
        if i == 0:
            term = f"{value:.{precision}f}"
        elif i == 1:
            term = f"{value:.{precision}f}*x"
        else:
            term = f"{value:.{precision}f}*x^{i}"
        result += sign + term
    print(result)


def solve_system(A, b):
    """Решение СЛАУ методом Гаусса с выбором главного элемента"""
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    n = len(b)

    for k in range(n):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[max_row, k]):
                max_row = i

        A[[k, max_row]] = A[[max_row, k]]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

    result = np.zeros(n)
    for i in range(n - 1, -1, -1):
        result[i] = (b[i] - np.sum(A[i, i + 1 :] * result[i + 1 :])) / A[i, i]
    return result


def divided_differences(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table


def polynomial_product(first, second):
    result = np.zeros(len(first) + len(second) - 1)
    for i in range(len(first)):
        for j in range(len(second)):
            result[i + j] += first[i] * second[j]
    return result
