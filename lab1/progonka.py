import numpy as np


def progonka(a, b, c, d):
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)

    if n == 1:
        Q[0] = d[0] / b[0]
        return P, Q, Q.copy()
    
    # Прямой ход.
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n - 1):
        den = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / den
        Q[i] = (d[i] - a[i] * Q[i - 1]) / den
    den = b[n - 1] + a[n - 1] * P[n - 2]
    P[n - 1] = 0.0
    Q[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / den

    # Обратный ход.
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return P, Q, x


def main():
    with open("matrix1.2.txt") as f:
        data = [[float(t) for t in line.split()] for line in f if line.strip()]
    M = np.array(data, dtype=float)
    n = M.shape[0]
    a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), M[:, -1]

    for i in range(n):
        b[i] = M[i, i]
        if i > 0:
            a[i] = M[i, i - 1]
        if i < n - 1:
            c[i] = M[i, i + 1]

    digits = 3

    P, Q, x = progonka(a, b, c, d)

    print("Прогоночные коэффициенты P_i:")
    print(np.round(P.reshape(n,1), digits), "\n")
    print("Прогоночные коэффициенты Q_i:")
    print(np.round(Q.reshape(n, 1), digits), "\n")

    print("Решение системы:")
    print(np.round(x.reshape(n, 1), digits), "\n")




if __name__ == "__main__":
    main()
