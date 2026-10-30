import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sqrt(1 - x ** 2) - np.exp(x) + 0.1


def df(x):
    return -x / np.sqrt(1 - x ** 2) - np.exp(x)


def d2f(x):
    return -1 / (1 - x ** 2) ** 1.5 - np.exp(x)


def phi(x):
    return np.log(np.sqrt(1 - x ** 2) + 0.1)


def dphi(x):
    return -x / np.sqrt(1 - x ** 2) / (np.sqrt(1 - x ** 2) + 0.1)


def newton(x0, epsilon, max_iter=1000):
    xs = [x0]
    for k in range(max_iter):
        x_new = xs[-1] - f(xs[-1]) / df(xs[-1])
        xs.append(x_new)
        if abs(x_new - xs[-2]) < epsilon:
            break
    return xs


def simple_iteration(x0, q, epsilon, max_iter=1000):
    xs = [x0]
    estimates = []
    for k in range(max_iter):
        x_new = phi(xs[-1])
        estimate = q / (1 - q) * abs(x_new - xs[-1])
        xs.append(x_new)
        estimates.append(estimate)
        if estimate <= epsilon:
            break
    return xs, estimates


def draw_graph(a, b, x0):
    x = np.linspace(a - 0.3, b + 0.3, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(x, f(x), label="f(x) = sqrt(1 - x^2) - e^x + 0.1", color="blue")
    plt.axhline(0, color="black", linestyle="--")
    plt.axvspan(a, b, color="green", alpha=0.2, label="Отрезок [a, b]")
    plt.plot(x0, f(x0), "ro", label="Начальное приближение x0")
    plt.title("Графическое отделение корня")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    with open("data2.1.txt") as file:
        data = [[float(t) for t in line.split()] for line in file if line.strip()]
    epsilon = data[0][0]
    a, b = data[1]
    x0 = data[2][0]

    digits = 10

    x_exact = newton(x0, 1e-14)[-1]
    x_grid = np.linspace(a, b, 101)
    q = np.max(np.abs(dphi(x_grid)))
    phi_inside = np.all(phi(x_grid) >= a) and np.all(phi(x_grid) <= b)

    print("Заданная точность epsilon:")
    print(epsilon, "\n")
    print("Отрезок [a, b], найденный по графику:")
    print(a, b, "\n")
    print("Начальное приближение x0:")
    print(x0, "\n")
    print("Эталонное значение корня:")
    print(np.round(x_exact, digits), "\n")

    print("Проверка условий метода Ньютона")
    print("f(a) * f(b) < 0:", f(a) * f(b) < 0)
    print("f(x0) * f''(x0) > 0:", f(x0) * d2f(x0) > 0, "\n")
    print("Проверка условий метода простой итерации")
    print("phi(x) лежит в [a, b]:", phi_inside)
    print("q = max|phi'(x)|:", np.round(q, digits), "\n")

    draw_graph(a, b, x0)

    xs, estimates = simple_iteration(x0, q, epsilon)
    print("Метод простой итерации")
    print("k, x(k), phi(x(k)), q/(1-q)*|x(k+1) - x(k)|, |x(k+1) - x*|")
    for k in range(len(estimates)):
        x = xs[k]
        x_next = xs[k + 1]
        error = abs(x_next - x_exact)
        print(k, np.round(x, digits), np.round(x_next, digits), np.round(estimates[k], digits), np.round(error, digits))
    print()
    print("Корень:")
    print(np.round(xs[-1], digits), "\n")
    print("Количество итераций:")
    print(len(estimates), "\n")

    xs = newton(x0, epsilon)
    print("Метод Ньютона")
    print("k, x(k), f(x(k)), f'(x(k)), |x(k+1) - x(k)|, |x(k+1) - x*|")
    for k in range(len(xs) - 1):
        x = xs[k]
        x_next = xs[k + 1]
        error = abs(x_next - x_exact)
        print(k, np.round(x, digits), np.round(f(x), digits), np.round(df(x), digits), np.round(abs(x_next - x), digits), np.round(error, digits))
    print()
    print("Корень:")
    print(np.round(xs[-1], digits), "\n")
    print("Количество итераций:")
    print(len(xs) - 1, "\n")

    print("Зависимость количества итераций от точности")
    for e in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        count_simple = len(simple_iteration(x0, q, e)[1])
        count_newton = len(newton(x0, e)) - 1
        print("epsilon =", e, "| простая итерация:", count_simple, "| Ньютон:", count_newton)


if __name__ == "__main__":
    main()