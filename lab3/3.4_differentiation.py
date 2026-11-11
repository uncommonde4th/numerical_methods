import numpy as np
from tools import read_numbers


def find_node(x, value):
    """Номер узла, совпадающего с value, или -1"""
    for i in range(len(x)):
        if abs(x[i] - value) < 1e-12:
            return i
    return -1


def find_segment(x, value):
    for i in range(len(x) - 1):
        if x[i] <= value <= x[i + 1]:
            return i
    raise ValueError("Точка x* лежит вне таблицы")


def one_sided(x, y, i):
    """Производная с первым порядком точности по узлам i, i+1: f(x_i, x_(i+1))"""
    return (y[i + 1] - y[i]) / (x[i + 1] - x[i])


def three_points(x, y, i, value):
    """Первая производная со вторым порядком точности и вторая производная по узлам i, i+1, i+2"""
    f01 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    f12 = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1])
    f012 = (f12 - f01) / (x[i + 2] - x[i])
    first = f01 + f012 * (2 * value - x[i] - x[i + 1])
    second = 2 * f012
    return first, second


def main():
    data = read_numbers("data_34.txt")
    n = int(data[0])
    x = data[1 : 1 + n]
    y = data[1 + n : 1 + 2 * n]
    x_star = data[1 + 2 * n]
    precision = int(data[2 + 2 * n])

    print("\nИсходная таблица:")
    for i in range(n):
        print(f"x[{i}] = {x[i]:.{precision}f}, y[{i}] = {y[i]:.{precision}f}")

    print(f"\nТочка вычисления: x* = {x_star:.{precision}f}")

    node = find_node(x, x_star)
    if node == -1:
        i = find_segment(x, x_star)
        print(f"x* лежит внутри отрезка [x[{i}], x[{i + 1}]], не совпадает с узлом")
    else:
        i = node - 1
        print(f"x* совпадает с узлом x[{node}]")
    i = max(0, min(i, n - 3))

    if node != -1:
        if node > 0:
            left = one_sided(x, y, node - 1)
            print(
                f"\nЛевосторонняя производная (узлы x[{node - 1}] = {x[node - 1]:g}, "
                f"x[{node}] = {x[node]:g}):"
            )
            print(f"f'(x*) = {left:.{precision}f}")
        if node < n - 1:
            right = one_sided(x, y, node)
            print(
                f"\nПравосторонняя производная (узлы x[{node}] = {x[node]:g}, "
                f"x[{node + 1}] = {x[node + 1]:g}):"
            )
            print(f"f'(x*) = {right:.{precision}f}")

    first, second = three_points(x, y, i, x_star)
    nodes_text = f"x[{i}] = {x[i]:g}, x[{i + 1}] = {x[i + 1]:g}, x[{i + 2}] = {x[i + 2]:g}"
    print(f"\nПервая производная со вторым порядком точности (узлы {nodes_text}):")
    print(f"f'(x*) = {first:.{precision}f}")
    print(f"\nВторая производная (узлы {nodes_text}):")
    print(f"f''(x*) = {second:.{precision}f}")


if __name__ == "__main__":
    main()