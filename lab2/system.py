import numpy as np
import matplotlib.pyplot as plt


def f(x, a):
    f1 = (x[0] ** 2 + a ** 2) * x[1] - a ** 3
    f2 = (x[0] - a / 2) ** 2 + (x[1] - a / 2) ** 2 - a ** 2
    return np.array([f1, f2])


def J(x, a):
    return np.array([[2 * x[0] * x[1], x[0] ** 2 + a ** 2],
                     [2 * (x[0] - a / 2), 2 * (x[1] - a / 2)]])


def phi(x, a):
    phi1 = a / 2 + np.sqrt(a ** 2 - (x[1] - a / 2) ** 2)
    phi2 = a ** 3 / (x[0] ** 2 + a ** 2)
    return np.array([phi1, phi2])


def dphi(x, a):
    return np.array([[0, -(x[1] - a / 2) / np.sqrt(a ** 2 - (x[1] - a / 2) ** 2)],
                     [-2 * a ** 3 * x[0] / (x[0] ** 2 + a ** 2) ** 2, 0]])


def norm(x):
    return np.max(np.abs(x))


def estimate_q(a, x1_range, x2_range):
    q = 0
    for x1 in np.linspace(x1_range[0], x1_range[1], 50):
        for x2 in np.linspace(x2_range[0], x2_range[1], 50):
            row_sums = np.sum(np.abs(dphi([x1, x2], a)), axis=1)
            q = max(q, np.max(row_sums))
    return q


def newton(a, x0, epsilon, max_iter=100):
    xs = [x0]
    dets = []
    for k in range(max_iter):
        x = xs[-1]
        fx = f(x, a)
        Jx = J(x, a)
        A1 = np.array([[fx[0], Jx[0, 1]], [fx[1], Jx[1, 1]]])
        A2 = np.array([[Jx[0, 0], fx[0]], [Jx[1, 0], fx[1]]])
        detJ = np.linalg.det(Jx)
        if abs(detJ) < 1e-12:
            print("Матрица Якоби вырождена")
            break
        x_new = x - np.array([np.linalg.det(A1), np.linalg.det(A2)]) / detJ
        xs.append(x_new)
        dets.append(detJ)
        if norm(x_new - x) <= epsilon:
            break
    return xs, dets


def simple_iteration(a, x0, q, epsilon, max_iter=100):
    xs = [x0]
    estimates = []
    for k in range(max_iter):
        x_new = phi(xs[-1], a)
        estimate = q / (1 - q) * norm(x_new - xs[-1])
        xs.append(x_new)
        estimates.append(estimate)
        if estimate <= epsilon:
            break
    return xs, estimates


def draw_graph(a, x0, x_exact):
    x1 = np.linspace(-a / 2, 3 * a / 2, 400)
    root = np.sqrt(np.maximum(a ** 2 - (x1 - a / 2) ** 2, 0))
    plt.figure(figsize=(8, 6))
    plt.plot(x1, a ** 3 / (x1 ** 2 + a ** 2), "b", label="f1 = 0: x2 = a^3 / (x1^2 + a^2)")
    plt.plot(x1, a / 2 + root, "r", label="f2 = 0: окружность")
    plt.plot(x1, a / 2 - root, "r")
    plt.plot(x0[0], x0[1], "ko", label="Начальное приближение")
    plt.plot(x_exact[0], x_exact[1], "g*", markersize=12, label="Решение")
    plt.axis("equal")
    plt.title("Графическое решение системы")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def main():
    with open("data2.2.txt") as file:
        data = [[float(t) for t in line.split()] for line in file if line.strip()]
    epsilon = data[0][0]
    a = data[1][0]
    x0 = np.array(data[2])
    x1_range = data[3]
    x2_range = data[4]

    digits = 10

    x_exact = newton(a, x0, 1e-14)[0][-1]
    q = estimate_q(a, x1_range, x2_range)

    print("Заданная точность epsilon:")
    print(epsilon, "\n")
    print("Параметр a:")
    print(a, "\n")
    print("Начальное приближение, найденное по графику:")
    print(x0, "\n")
    print("Область G: x1 от", x1_range[0], "до", x1_range[1], ", x2 от", x2_range[0], "до", x2_range[1], "\n")
    print("Эталонное решение:")
    print(np.round(x_exact, digits), "\n")
    print("q = max||phi'(x)|| на области G:")
    print(np.round(q, digits), "\n")
    if q >= 1:
        print("Условие сходимости q < 1 не выполнено", "\n")

    draw_graph(a, x0, x_exact)

    xs, estimates = simple_iteration(a, x0, q, epsilon)
    print("Метод простой итерации")
    print("k, x(k), phi(x(k)), q/(1-q)*||x(k+1) - x(k)||, ||x(k+1) - x*||")
    for k in range(len(estimates)):
        x = xs[k]
        x_next = xs[k + 1]
        error = norm(x_next - x_exact)
        print(k, np.round(x, digits), np.round(x_next, digits), np.round(estimates[k], digits), np.round(error, digits))
    print()
    inside = True
    for x in xs:
        if not (x1_range[0] <= x[0] <= x1_range[1] and x2_range[0] <= x[1] <= x2_range[1]):
            inside = False
    print("Приближения не выходят из области G:", inside, "\n")
    print("Решение:")
    print(np.round(xs[-1], digits), "\n")
    print("Количество итераций:")
    print(len(estimates), "\n")

    xs, dets = newton(a, x0, epsilon)
    print("Метод Ньютона")
    print("k, x(k), f(x(k)), det J, ||x(k+1) - x(k)||, ||x(k+1) - x*||")
    for k in range(len(dets)):
        x = xs[k]
        x_next = xs[k + 1]
        error = norm(x_next - x_exact)
        print(k, np.round(x, digits), np.round(f(x, a), digits), np.round(dets[k], digits), np.round(norm(x_next - x), digits), np.round(error, digits))
    print()
    print("Решение:")
    print(np.round(xs[-1], digits), "\n")
    print("Количество итераций:")
    print(len(dets), "\n")

    print("Зависимость количества итераций от точности")
    for e in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        count_simple = len(simple_iteration(a, x0, q, e)[1])
        count_newton = len(newton(a, x0, e)[1])
        print("epsilon =", e, "| простая итерация:", count_simple, "| Ньютон:", count_newton)


if __name__ == "__main__":
    main()