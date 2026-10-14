import numpy as np


def sign(x):
    if x >= 0:
        return 1.0
    return -1.0


def column_norm(a, j, start):
    s = 0.0
    for l in range(start, a.shape[0]):
        s += a[l, j] ** 2
    return s ** 0.5


def QR(a, n):
    q = np.eye(n)
    for i in range(n - 1):
        norm = column_norm(a, i, i)
        if norm == 0.0:
            continue

        v = np.zeros(n)
        v[i] = a[i, i] + sign(a[i, i]) * norm
        for j in range(i + 1, n):
            v[j] = a[j, i]

        v = v.reshape(n, 1)
        hi = np.eye(n) - 2 * ((v @ v.transpose()) / (v.transpose() @ v))
        a = hi @ a
        q = q @ hi
    return q, a


def block_eigenvalues(a, j):
    p = a[j, j] + a[j + 1, j + 1]
    q = a[j, j] * a[j + 1, j + 1] - a[j, j + 1] * a[j + 1, j]
    D = p * p - 4 * q
    if D >= 0:
        s = D ** 0.5
        return (p + s) / 2, (p - s) / 2
    s = (-D) ** 0.5
    return complex(p / 2, s / 2), complex(p / 2, -s / 2)


def get_eigenvalues(a, epsilon, prev):
    n = a.shape[0]
    eig = [0] * n
    blocks = [None] * n
    d = 0.0
    j = 0
    while j < n:
        col = column_norm(a, j, j + 1)
        if col <= epsilon:
            eig[j] = a[j, j]
            j += 1
            continue

        under = max(column_norm(a, j, j + 2), column_norm(a, j + 1, j + 2))
        if under > epsilon:
            eig[j] = a[j, j]
            d = max(d, col)
            j += 1
            continue

        h1, h2 = block_eigenvalues(a, j)
        eig[j] = h1
        eig[j + 1] = h2
        blocks[j] = h1
        blocks[j + 1] = h2
        if prev[j] is None or prev[j + 1] is None:
            d = max(d, 1e9)
        else:
            d = max(d, abs(h1 - prev[j]), abs(h2 - prev[j + 1]))
        j += 2
    return eig, blocks, d


def qr_algorithm(a0, epsilon, maxiter):
    n = a0.shape[0]
    a = a0.copy()
    eig = [0] * n
    blocks = [None] * n
    d = 1e9
    k = 0
    while d > epsilon and k < maxiter:
        k += 1
        q, r = QR(a, n)
        a = r @ q
        eig, blocks, d = get_eigenvalues(a, epsilon, blocks)
    return a, eig, d, k


def main():
    with open("matrix1.5.txt") as f:
        data = [[float(t) for t in line.split()] for line in f if line.strip()]
    a0 = np.array(data, dtype=float)
    n = a0.shape[0]

    epsilon = 0.01
    maxiter = 5000
    digits = 3

    q0, r0 = QR(a0, n)
    a, eig, d, k = qr_algorithm(a0, epsilon, maxiter)

    print("Заданная точность epsilon:")
    print(epsilon, "\n")

    print("Исходная матрица A:")
    print(np.round(a0, digits), "\n")
    print("Матрица Q:")
    print(np.round(q0, digits), "\n")
    print("Матрица R:")
    print(np.round(r0, digits), "\n")
    print("Q * R:")
    print(np.round(q0 @ r0, digits), "\n")

    print("Количество итераций:")
    print(k, "\n")
    if d > epsilon:
        print("Точность за", maxiter, "итераций не достигнута\n")

    print("Матрица A после итераций:")
    print(np.round(a, digits), "\n")

    print("Собственные значения:")
    for i in range(n):
        print(f"λ{i + 1} :", np.round(eig[i], digits))
    print()

    print("Проверка:")
    print(np.round(np.linalg.eigvals(a0), digits), "\n")


if __name__ == "__main__":
    main()