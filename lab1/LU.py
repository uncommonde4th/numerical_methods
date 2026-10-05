import numpy as np


def lu_decompose(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0

    for k in range(n - 1):
        # Выбираем главный элемент.
        m = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[m, k]):
                m = i
        if abs(U[m, k]) < 1e-12:
            raise ValueError("Матрица вырождена")

        # Перестановка строк k и m.
        if m != k:
            tmp = U[k].copy()
            U[k] = U[m]
            U[m] = tmp

            tmp = P[k].copy()
            P[k] = P[m]
            P[m] = tmp

            for j in range(k):
                tmp = L[k, j]
                L[k, j] = L[m, j]
                L[m, j] = tmp
            swaps += 1

        # Обнуление поддиагональных элементов k-го столбца.
        for i in range(k + 1, n):
            mu = U[i, k] / U[k,k]
            L[i, k] = mu
            for j in range(k, n):
                U[i, j] = U[i, j] - mu * U[k, j]
        
    if abs(U[n - 1, n - 1]) < 1e-12:
        raise ValueError("Матрица вырождена")
    
    return L, U, P, swaps


def solve_lu(L, U, P, b):
    n = L.shape[0]
    Pb = P @ b

    # Lz = Pb
    z = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * z[j]
        z[i] = Pb[i] - s
    
    # Ux = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (z[i] - s) / U[i, i]
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
