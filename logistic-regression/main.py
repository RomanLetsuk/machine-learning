import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


FILENAME = './data/ex2data1.txt'


def load_data():
    return pd.read_csv(FILENAME, header=None, names=['first_mark', 'second_mark', 'entered'])


df = load_data()
X, y = df.filter(['first_mark', 'second_mark']), df['entered']


def show_data():
    entered = df[df['entered'] == 1]
    not_entered = df[df['entered'] == 0]
    fig, ax = plt.subplots()
    ax.scatter(entered['first_mark'], entered['second_mark'], marker='o', color='b', s=20)
    ax.scatter(not_entered['first_mark'], not_entered['second_mark'], marker='x', color='r')
    ax.set_title('Entered')
    ax.set_xlabel('First mark')
    ax.set_ylabel('Second mark')
    ax.grid(True)

    plt.show()


def sigmoid_function(z):
    return 1 / (1 + np.e ** -z)


class LogisticRegression:
    THRESHOLD = 1e-6

    def __init__(self, learning_rate=0.003, max_steps=300000, method='gradient_descent'):
        self.weights = []
        self.costs = []
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.method = method

    def set_data(self, X, y):
        X = X.to_numpy().astype('float64')
        y = y.to_numpy().astype('float64')
        self.weights = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        optimization_method = getattr(self, self.method)
        optimization_method(X, y)

    def calculate_hypothesis(self, X, weights=None):
        if weights is None:
            weights = self.weights

        return sigmoid_function(X.dot(weights))

    def calculate_cost_function(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        predictions = self.calculate_hypothesis(X, weights)
        trues = y * np.log(predictions)
        falses = (1 - y) * np.log(1 - predictions)
        return -np.mean(trues + falses)

    def gradien_descent_step(self, X, y):
        delta = self.calculate_derivative(X, y)
        delta *= self.learning_rate
        self.weights -= delta

    def gradient_descent(self, X, y):
        current_cost = self.calculate_cost_function(X, y)
        current_step = 0
        self.costs.append(current_cost)

        while current_step < self.max_steps:
            current_step += 1
            self.gradien_descent_step(X, y)
            new_cost = self.calculate_cost_function(X, y)
            self.costs.append(new_cost)
            if abs(new_cost - current_cost) < self.THRESHOLD:
                break

            current_cost = new_cost

    def calculate_derivative(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        predictions = self.calculate_hypothesis(X, weights)
        diff = predictions - y
        return np.dot(X.T, diff) / X.shape[0]

    def nelder_mead(self, X, y):
        def evaluate(weights):
            return self.calculate_cost_function(X, y, weights)

        init_weights = np.zeros(X.shape[1])
        self.weights= optimize.fmin(evaluate, init_weights, xtol=self.THRESHOLD, maxfun=self.max_steps)

    def bfgs(self, X, y):
        def evaluate(weights):
            return self.calculate_cost_function(X, y, weights)

        def evaluate_derivative(weights):
            return self.calculate_derivative(X, y, weights)

        init_weights = np.zeros(X.shape[1])
        self.weights = optimize.fmin_bfgs(evaluate, init_weights, evaluate_derivative, gtol=self.THRESHOLD)

    def predict(self, marks):
        marks = np.column_stack((np.ones(marks.shape[0]), marks))
        predictions = self.calculate_hypothesis(marks)
        return list(map(lambda prediction: 1 if prediction > 0.5 else 0, predictions))


    def show_boundary(self):
        entered = df[df['entered'] == 1]
        not_entered = df[df['entered'] == 0]
        fig, ax = plt.subplots()
        ax.scatter(entered['first_mark'], entered['second_mark'], marker='o', color='b', s=20)
        ax.scatter(not_entered['first_mark'], not_entered['second_mark'], marker='x', color='r')
        x = np.linspace(np.min(df['first_mark'] - 3), np.max(df['first_mark'] + 3))
        ax.plot(x, -(self.weights[0] + self.weights[1] * x) / self.weights[2])
        ax.set_title('Entered')
        ax.set_xlabel('First mark')
        ax.set_ylabel('Second mark')
        ax.grid(True)

        plt.show()


def main():
    # show_data()
    lr = LogisticRegression()
    lr.set_data(X, y)
    # lr2 = LogisticRegression(method='nelder_mead')
    # lr2.set_data(X, y)
    # lr3 = LogisticRegression(method='bfgs')
    # lr3.set_data(X, y)
    print(lr.predict(np.array([[40, 40], [50, 50], [60, 60], [70, 70], [70, 60], [60, 70], [35, 90]])))
    lr.show_boundary()


if __name__ == '__main__':
    main()
