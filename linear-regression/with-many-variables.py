import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

FILENAME = './data/ex1data2.txt'


class LinearRegression:
    THRESHOLD = 0.001

    def __init__(self, learning_rate=0.01, max_steps=100000, normalized=True, vectorized=True, normal_equation=False):
        self.weights = []
        self.costs = []
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.normalized = normalized
        self.vectorized = vectorized
        self.normal_equation = normal_equation


    def set_data(self, X, y):
        X = X.to_numpy().astype('float64')
        y = y.to_numpy().astype('float64')
        self.weights = np.zeros(X.shape[1] + 1)
        if self.normalized:
            X = self.normalize(X)
        X = np.column_stack((np.ones(X.shape[0]), X))
        if self.normal_equation:
            self.calc_normal_equation(X, y)
        else:
            self.gradient_descent(X, y)


    def normalize(self, X):
        normalized_X = X.copy()
        for i in range(X.shape[1]):
            feature = X[:, i]
            delta = np.max(feature) - np.min(feature)
            normalized_X[:, i] = (feature - np.mean(feature)) / delta
        return normalized_X


    def get_predictions(self, X):
        return X.dot(self.weights)


    def cost_function(self, X, y):
        predictions = self.get_predictions(X)
        return np.mean((predictions - y) ** 2) / 2

    def gradient_descent(self, X, y):
        current_cost = self.cost_function(X, y)
        current_step = 0
        self.costs.append(current_cost)

        while current_step < self.max_steps:
            current_step += 1
            self.gradient_descent_step(X, y)
            new_cost = self.cost_function(X, y)
            self.costs.append(new_cost)
            if abs(new_cost - current_cost) < self.THRESHOLD:
                break

            current_cost = new_cost


    def gradient_descent_step(self, X, y):
        predictions = self.get_predictions(X)
        diff = predictions - y
        if self.vectorized:
            self.gradient_descent_step_vectorized(X, diff)
        else:
            self.gradient_descent_step_simple(X, diff)

    def gradient_descent_step_vectorized(self, X, diff):
        delta = np.dot(X.T, diff)
        delta *= (self.learning_rate / X.shape[0])
        self.weights -= delta


    def gradient_descent_step_simple(self, X, diff):
        count = X.shape[0]
        for i in range(X.shape[1]):
            delta = np.mean(X[:, i] * diff)
            self.weights[i] -= (delta * self.learning_rate)

    def calc_normal_equation(self, X, y):
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
        self.costs.append(self.cost_function(X, y))


def load_data():
    df = pd.read_csv(FILENAME, header=None, names=['area', 'rooms', 'price'])
    return df.filter(['area', 'rooms']), df['price']


def show_normalization_result(not_normalized, normalized):
    fig1, ax1 = plt.subplots()
    ax1.plot(not_normalized)
    ax1.set_title('Not normalized')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cost Function')
    ax1.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(normalized)
    ax2.set_title('Normalized')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cost Function')
    ax2.grid(True)

    plt.show()


def normalization_difference(X, y):
    lr = LinearRegression(learning_rate=0.01, max_steps=100, normalized=False)
    lr.set_data(X, y)
    lr2 = LinearRegression(learning_rate=0.01, max_steps=100, normalized=True)
    lr2.set_data(X, y)
    show_normalization_result(lr.costs, lr2.costs)


def vectorization_difference(X, y):
    lr = LinearRegression(learning_rate=0.01, vectorized=False)
    lr2 = LinearRegression(learning_rate=0.01, vectorized=True)
    print('Without vectorization:')
    start_time = dt.now()
    lr.set_data(X, y)
    end_time = dt.now()
    print('execution time: ', end_time - start_time)
    print('With vectorization:')
    start_time = dt.now()
    lr2.set_data(X, y)
    end_time = dt.now()
    print('execution time: ', end_time - start_time)


def learning_rate_difference(X, y):
    lr = LinearRegression(learning_rate=1)
    lr.set_data(X, y)
    lr2 = LinearRegression(learning_rate=0.1)
    lr2.set_data(X, y)
    lr3 = LinearRegression(learning_rate=0.01)
    lr3.set_data(X, y)
    lr4 = LinearRegression(learning_rate=0.001)
    lr4.set_data(X, y)

    fig1, ax1 = plt.subplots()
    ax1.plot(lr.costs)
    ax1.set_title('Learning rate = 1')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cost Function')
    ax1.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(lr2.costs)
    ax2.set_title('Learning rate = 0.1')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cost Function')
    ax2.grid(True)

    fig3, ax3 = plt.subplots()
    ax3.plot(lr3.costs)
    ax3.set_title('Learning rate = 0.01')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Cost Function')
    ax3.grid(True)

    fig4, ax4 = plt.subplots()
    ax4.plot(lr4.costs)
    ax4.set_title('Learning rate = 0.001')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Cost Function')
    ax4.grid(True)

    plt.show()


def normal_equation_difference(X, y):
    lr = LinearRegression(normal_equation=True)
    lr.set_data(X, y)
    lr2 = LinearRegression(normal_equation=False)
    lr2.set_data(X, y)
    print('With normal equation:')
    print(lr.costs[-1])
    print('Without normal equation:')
    print(lr2.costs[-1])


def main():
    X, y = load_data()
    normalization_difference(X, y)
    vectorization_difference(X, y)
    learning_rate_difference(X, y)
    normal_equation_difference(X, y)


if __name__ == '__main__':
    main()