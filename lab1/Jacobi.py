import numpy as np


def off_diagonal_norm(A):
    n = A.shape[0]
    s = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            s += A[i, j] ** 2
    
    return np.sqrt(s)


def rotation_angle(aii, ajj, aij):
    if aii == ajj:
        return np.pi / 4
    return 0.5 * np.arctan(2.0 * aij / (aii - ajj))


def rotation_matrix(n, i, j, phi):
    U = np.eye(n)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    U[i, i] = cos_phi
    U[j, j] = cos_phi
    U[i, j] = -sin_phi
    U[j, i] = sin_phi
    return U


def jacobi(A, eps, max_iter=100000):
    n = A.shape[0]
    Ak = A.copy()
    V = np.eye(n)
    
    iterations = 0

    while off_diagonal_norm(Ak) >= eps:
        max_value = 0.0
        p = 0
        q = 1

        for i in range(n):
            for j in range(i + 1, n):
                if abs(Ak[i, j]) > max_value:
                    max_value = abs(Ak[i, j])
                    p = i
                    q = j

        phi = rotation_angle(Ak[p, p], Ak[q, q], Ak[p, q])

        U = rotation_matrix(n, p, q, phi)

        Ak = U.T @ Ak @ U
        V = V @ U
        iterations += 1

    values = np.zeros(n)

    for i in range(n):
        values[i] = Ak[i, i]

    return values, V, Ak, iterations


def main():
    with open("matrix1.4.txt") as f:
        data = [[float(t) for t in line.split()] for line in f if line.strip()]

    eps = 0.01
    A = np.array(data, dtype=float)

    n = A.shape[0]

    print("Точность эпсилон:")
    print(eps)

    values, V, Ak, iterations = jacobi(A, eps)

    print("Собственные значения:")
    print(np.round(values, 2))

    print("Собственные векторы:")
    for i in range(n):
        print(f"h{i + 1} :")
        print(np.round(V[0:n, i], 2))
    print("Матрица собственных векторов:")
    print(np.round(V, 2))
    print("A * V:")
    print(np.round(A @ V, 2))

    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = values[i]
    print("V * L:")
    print(np.round(V @ L, 2))


if __name__ == "__main__":
    main()
