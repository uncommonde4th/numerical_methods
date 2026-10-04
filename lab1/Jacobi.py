import numpy as np


def off_diagonal_norm(A):
    return float(np.sqrt(np.sum(np.triu(A, 1) ** 2)))


def rotation_angle(aii, ajj, aij):
    if aii == ajj:
        return np.pi / 4
    return 0.5 * np.arctan(2.0 * aij / (aii - ajj))


def rotation_matrix(n, i, j, phi):
    U = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -s
    U[j, i] = s
    return U


def jacobi(A, eps, max_iter=100000):
    n = A.shape[0]
    Ak = A.astype(float).copy()
    V = np.eye(n)
    log = []
    k = 0
    while off_diagonal_norm(Ak) >= eps and k < max_iter:
        upper = np.abs(np.triu(Ak, 1))
        i, j = 
