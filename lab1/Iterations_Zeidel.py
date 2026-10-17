import numpy as np


def build_alpha_beta(A, b):
    n = A.shape[0]
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]
    return alpha, beta


def matrix_norm(alpha):
    n = alpha.shape[0]
    result = 0
    for i in range(n):
        s = 0
        for j in range(n):
            s += abs(alpha[i, j])
        if s > result:
            result = s
    return result


def upper_norm(alpha):
    n = alpha.shape[0]
    result = 0
    for i in range(n):
        s = 0
        for j in range(i + 1, n):
            s += abs(alpha[i, j])
        if s > result:
            result = s
    return result


def max_diff(x_new, x_old):
    result = 0
    for i in range(len(x_new)):
        if abs(x_new[i] - x_old[i]) > result:
            result = abs(x_new[i] - x_old[i])
    return result


def simple_iterations(alpha, beta, epsilon):
    alpha_norm = matrix_norm(alpha)
    x = beta.copy()
    err = epsilon + 1
    count = 0
    while err > epsilon:
        x_new = alpha @ x + beta
        diff = max_diff(x_new, x)
        if alpha_norm < 1:
            err = alpha_norm / (1 - alpha_norm) * diff
        else:
            err = diff
        x = x_new
        count += 1
    return x, count, err


def seidel(alpha, beta, epsilon):
    n = alpha.shape[0]
    alpha_norm = matrix_norm(alpha)
    c_norm = upper_norm(alpha)
    x = beta.copy()
    err = epsilon + 1
    count = 0
    while err > epsilon:
        x_new = x.copy()
        for i in range(n):
            s = beta[i]
            for j in range(n):
                s += alpha[i, j] * x_new[j]
            x_new[i] = s
        diff = max_diff(x_new, x)
        if alpha_norm < 1:
            err = c_norm / (1 - alpha_norm) * diff
        else:
            err = diff
        x = x_new
        count += 1
    return x, count, err


def main():
    with open("matrix1.3.txt") as f:
        data = [[float(t) for t in line.split()] for line in f if line.strip()]
    M = np.array(data, dtype=float)
    n = M.shape[0]
    A, b = M[:, :n], M[:, n]

    epsilon = 0.01

    alpha, beta = build_alpha_beta(A, b)
    x1, count1, err1 = simple_iterations(alpha, beta, epsilon)
    x2, count2, err2 = seidel(alpha, beta, epsilon)

    print("Заданная точность epsilon:")
    print(epsilon, "\n")

    print("Точное решение X:")
    print(np.round(np.linalg.solve(A, b).reshape(n, 1), 4), "\n")

    print("Метод простых итераций X:")
    print(np.round(x1.reshape(n, 1), 4), "\n")
    print("Количество итераций:")
    print(count1, "\n")

    print("Метод Зейделя X:")
    print(np.round(x2.reshape(n, 1), 4), "\n")
    print("Количество итераций:")
    print(count2, "\n")

    if not err1 <= epsilon:
        print("Метод простых итераций не сошёлся")
    if not err2 <= epsilon:
        print("Метод Зейделя не сошёлся")

    if err1 <= epsilon and err2 <= epsilon:
        if count2 < count1:
            print("Метод Зейделя сходится быстрее")
        elif count1 < count2:
            print("Метод простых итераций сходится быстрее")
        else:
            print("Оба метода сходятся одинаково")


if __name__ == "__main__":
    main()
