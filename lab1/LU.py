import numpy as np


def lu_decompose(A):
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    perm = list(range(n))
    swaps = 0
    tol = 1e-12 * max(1.0, np.max(np.abs(A)))

    for k in range(n - 1):
        m = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[m, k]) < tol:
            raise ValueError("Матрица вырождена")
        if m != k:
            U[[k, m], :] = U[[m, k], :]
            L[[k, m], :k] = L[[m, k], :k]
            perm[k], perm[m] = perm[m], perm[k]
            swaps += 1

        for i in range(k + 1, n):
            mu = U[i, k] / U[k, k]
            L[i, k] = mu
            U[i, k:] = U[i, k:] - mu * U[k, k:]
            U[i, k] = 0.0
    if abs(U[n - 1, n - 1]) < tol:
        raise ValueError("Матрица вырождена: U_nn = 0")

    P = np.zeros((n, n))
    for i in range(n):
        P[i, perm[i]] = 1.0
    return L, U, P, swaps


def solve_lu(L, U, P, b):
    n = L.shape[0]
    Pb = P @ b
    z = np.zeros(n)
    for i in range(n):
        z[i] = Pb[i] - np.dot(L[i, :i], z[:i])
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def inverse_lu(L, U, P):
    n = L.shape[0]
    E = np.eye(n)
    inv = np.zeros((n, n))
    for j in range(n):
        inv[:, j] = solve_lu(L, U, P, E[:, j])
    return inv


def determinant_lu(U, swaps):
    return (-1) ** swaps * float(np.prod(np.diag(U)))


def main():
    with open("matrix1.1.txt") as f:
        data = [[float(t) for t in line.split()] for line in f if line.strip()]
    M = np.array(data, dtype=float)
    n = M.shape[0]
    A, b = M[:, :n], M[:, n]

    L, U, P, swaps = lu_decompose(A)

    epsilon = 3

    print("L матрица:")
    print(np.round(L, epsilon), "\n")
    print("U матрица:")
    print(np.round(U, epsilon), "\n")

    print("LU матрица:")
    print(np.round(L @ U, epsilon), "\n")

    print("Решение системы Ax = b:")
    print(np.round(solve_lu(L, U, P, b).reshape(n, 1), epsilon), "\n")

    inv = inverse_lu(L, U, P)
    print("Обратная матрица A⁻¹:")
    print(np.round(inv, epsilon), "\n")

    print("Определитель матрицы A:")
    print(np.round(determinant_lu(U, swaps), epsilon), "\n")

    print("Проверка A · A⁻¹ = E")
    print("A · A⁻¹ =\n", np.round(A @ inv, epsilon), "\n")

    print("Проверка L · U = P · A")
    print("LU матрица: \n", np.round(L @ U, epsilon), "\n")
    print("PA матрица: \n", np.round(P @ A, epsilon), "\n")


if __name__ == "__main__":
    main()
