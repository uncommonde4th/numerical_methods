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
        P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
        Q[i] = (d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])
    P[n - 1] = 0.0
    Q[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / (b[n - 1] + a[n - 1] * P[n - 2])

    # Обратный ход.
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return P, Q, x


def main():
    a, b, c, d = [], [], [], []
    with open("matrix1.2.txt") as f:
        for line in f:
            if not line.strip():
                continue
            row = [float(t) for t in line.split()]
            i = len(b)
            a.append(row[i - 1] if i > 0 else 0.0)
            b.append(row[i])
            c.append(row[i + 1] if i < len(row) - 2 else 0.0)
            d.append(row[-1])
    a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)

    n = len(b)
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
