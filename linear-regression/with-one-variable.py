import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.colors
import mpl_toolkits.mplot3d
import numpy as np
from functools import reduce
from math import fabs

profit_data_path = './data/ex1data1.txt'
learning_rate = 0.01


def load_profit_data():
    profit_data = pd.read_csv(profit_data_path, ',', header=None)
    return np.array(profit_data), np.transpose(profit_data.values)


def show_profit_from_population(data):
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.grid(True)
    plt.plot(data[0], data[1], 'ro')
    plt.show()


def cost_func(data, theta_0, theta_1):
    return reduce(lambda result, row: result + ((theta_0 + theta_1 * row[0]) - row[1]) ** 2, data, 0) / (2 * len(data))


def gradient_descent(data, theta_0, theta_1):
    delta = 0.001
    count = len(data)
    while True:
        temp0 = theta_0 - learning_rate / count * reduce(lambda result, row: result + theta_0 + theta_1 * row[0] - row[1], data, 0)
        temp1 = theta_1 - learning_rate / count * reduce(lambda result, row: result + (theta_0 + theta_1 * row[0] - row[1]) * row[0], data, 0)
        # print(temp0, temp1, cost_func(data, temp0, temp1))
        if fabs(temp0 - theta_0) < delta and fabs(temp1 - theta_1) < delta:
            break
        if fabs(temp0 - theta_0) > 0.001:
            theta_0 = temp0
        if fabs(temp1 - theta_1) > 0.001:
            theta_1 = temp1
    # plt.show()
    return theta_0, theta_1


def show_hypothesis_func(data, theta_0, theta_1):
    x = np.linspace(0, 25)
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.grid(True)
    plt.plot(data[0], data[1], 'ro')
    plt.plot(x, theta_0 + theta_1 * x)
    plt.show()


def show_3d_surface(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-4, -2, 0.05)
    y = np.arange(0, 2, 0.05)
    x, y = np.meshgrid(x, y)
    z = np.array([cost_func(data, x, y) for x, y in zip(x, y)])
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('theta 0')
    ax.set_ylabel('theta 1')
    ax.set_zlabel('cost')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig, ax = plt.subplots()
    CS = ax.contour(x, y, z, levels=30)
    ax.clabel(CS, inline=1, fontsize=5)
    plt.show()


def main():
    profit_data, transposed_profit_data = load_profit_data()
    # print(transposed_profit_data)
    # show_profit_from_population(transposed_profit_data)
    # print(cost_func(profit_data, -3.2953852026551806, 1.1316717519613395))
    theta_0, theta_1 = gradient_descent(profit_data, theta_0=0, theta_1=0)
    print('result', 'theta0 =', theta_0, 'theta1 =', theta_1)
    show_hypothesis_func(transposed_profit_data, theta_0, theta_1)
    show_3d_surface(transposed_profit_data)


if __name__ == '__main__':
    main()
