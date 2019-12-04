import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize, io

FILENAME = './data/ex2data1.txt'


def load_data(filename, names):
    return pd.read_csv(filename, header=None, names=names)


df = load_data('./data/ex2data1.txt', ['first_mark', 'second_mark', 'entered'])
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

    def __init__(self, learning_rate=0.003, max_steps=300000, method='gradient_descent', regularized=False, regularization_parameter=0.5):
        self.weights = []
        self.costs = []
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.method = method
        self.regularized = regularized
        self.regularization_parameter = regularization_parameter

    def set_data(self, X, y):
        X = X.astype('float64')
        y = y.astype('float64')
        if not self.regularized:
            X = np.column_stack((np.ones(X.shape[0]), X))
        self.weights = np.zeros(X.shape[1])
        optimization_method = getattr(self, self.method)
        optimization_method(X, y)

    def calculate_hypothesis(self, X, weights=None):
        if weights is None:
            weights = self.weights

        return sigmoid_function(X.dot(weights))

    def calculate_cost_function(self, X, y, weights=None):
        return self.calculate_cost_function_regularized(X, y, weights) if self.regularized else self.calculate_cost_function_not_regularized(X, y, weights)

    def calculate_cost_function_not_regularized(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        predictions = self.calculate_hypothesis(X, weights)
        trues = y * np.log(predictions)
        falses = (1 - y) * np.log(1 - predictions)
        return -np.mean(trues + falses)

    def calculate_cost_function_regularized(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        cost = self.calculate_cost_function_not_regularized(X, y, weights)
        regularized_weights = weights[1:]
        regularized_cost = self.regularization_parameter / 2 / X.shape[0] * np.dot(regularized_weights.T, regularized_weights)
        return cost + regularized_cost

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
        return self.calculate_derivative_regularized(X, y, weights) if self.regularized else self.calculate_cost_function_not_regularized(X, y, weights)

    def calculate_derivative_not_regularized(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        predictions = self.calculate_hypothesis(X, weights)
        diff = predictions - y
        return np.dot(X.T, diff) / X.shape[0]

    def calculate_derivative_regularized(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        new_weights = self.calculate_derivative_not_regularized(X, y, weights)
        new_weights_regulirized = (self.regularization_parameter / X.shape[0]) * new_weights[1:]
        new_weights_regulirized = np.insert(new_weights_regulirized, 0, 0)
        return new_weights + new_weights_regulirized

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

    def predict(self, X):
        if not self.regularized:
            X = np.column_stack((np.ones(X.shape[0]), X))
        predictions = self.calculate_hypothesis(X)
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


# show_data()
# lr = LogisticRegression()
# lr.set_data(X, y)
# lr2 = LogisticRegression(method='nelder_mead')
# # lr2.set_data(X, y)
# # lr3 = LogisticRegression(method='bfgs')
# # lr3.set_data(X, y)
# print(lr.predict(np.array([[40, 40], [50, 50], [60, 60], [70, 70], [70, 60], [60, 70], [35, 90]])))
# lr.show_boundary()


def show_tests_data():
    passed = df[df['passed'] == 1]
    not_passed = df[df['passed'] == 0]
    fig, ax = plt.subplots()
    ax.scatter(passed['first_test'], passed['second_test'], marker='o', color='b')
    ax.scatter(not_passed['first_test'], not_passed['second_test'], marker='x', color='r')
    ax.set_title('Passed')
    ax.set_xlabel('First test')
    ax.set_ylabel('Second test')
    ax.grid(True)

    plt.show()


df = load_data('./data/ex2data2.txt', ['first_test', 'second_test', 'passed'])
X, y = df.filter(['first_test', 'second_test']), df['passed']

# show_tests_data()

def build_poly(x1, x2):
    degree = 6
    result = []
    for i in range(degree + 1):
        for j in range(i, degree + 1):
            result.append(x1 ** i * x2 ** (j - i))
    assert len(result) == 28
    return np.array(result).T


def show_poly_boundary(gr, nm, bfgs):
    x1 = np.linspace(-1, 1.2)
    x2 = np.linspace(-1, 1.2)
    z1 = np.zeros(shape=(len(x1), len(x2)))
    z2 = np.zeros(shape=(len(x1), len(x2)))
    z3 = np.zeros(shape=(len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            z1[i, j] = np.dot(build_poly(x1[i], x2[j]), gr.weights)
            z2[i, j] = np.dot(build_poly(x1[i], x2[j]), nm.weights)
            z3[i, j] = np.dot(build_poly(x1[i], x2[j]), bfgs.weights)

    fig, ax_reg = plt.subplots()
    ax_reg.contour(x1, x2, z1, levels=0, colors='r')
    ax_reg.contour(x1, x2, z2, levels=0, colors='g')
    ax_reg.contour(x1, x2, z3, levels=0, colors='y')
    trues = df[df['passed'] == 1]
    falses = df[df['passed'] == 0]
    ax_reg.scatter(trues['first_test'], trues['second_test'], marker='o', c='b')
    ax_reg.scatter(falses['first_test'], falses['second_test'], marker='x', c='r')
    ax_reg.set_ylabel('Second test')
    plt.show()


X_poly = build_poly(X['first_test'], X['second_test'])
# lr_poly = LogisticRegression(regularized=True)
# lr_poly.set_data(X_poly, y)
# print(lr_poly.weights)
# print(lr_poly.costs[-1])
# lr_poly2 = LogisticRegression(regularized=True, method='nelder_mead')
# lr_poly2.set_data(X_poly, y)
# print(lr_poly2.weights)
# lr_poly3 = LogisticRegression(regularized=True, method='bfgs')
# lr_poly3.set_data(X_poly, y)
# print(lr_poly3.weights)
# print(lr_poly3.predict(np.array(X_poly)))
# print(list(y))
# show_poly_boundary(lr_poly, lr_poly2, lr_poly3)

# lr1 = LogisticRegression(regularized=True, regularization_parameter=0.5)
# lr1.set_data(X_poly, y)
# lr2 = LogisticRegression(regularized=True, regularization_parameter=0.05)
# lr2.set_data(X_poly, y)
# lr3 = LogisticRegression(regularized=True, regularization_parameter=0.005)
# lr3.set_data(X_poly, y)
# show_poly_boundary(lr1, lr2, lr3)

def get_image(vector):
    return np.flip(np.reshape(vector, (20, 20)).T, axis=0)

mat_data = io.loadmat('./data/ex2data3.mat')
X, y = mat_data['X'], mat_data['y']
images = [get_image(i) for i in X]
y = np.where(y == 10, 0, y)
y = y.reshape(y.shape[0])
indeces = [np.where(y == i)[0][0] for i in np.unique(y)]

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axs.flatten()):
    ax.pcolor(images[indeces[i]], cmap=cm.gray)
    res = y[indeces[i]]
    ax.set_title(f'Number {res}')

plt.show()

class MulticlassLogisticRegression:
    def __init__(self, classes=10):
        self.classes = classes
        self.classifiers = [
            LogisticRegression(learning_rate=0.5, regularized=True, regularization_parameter=0.1)
            for i in range(classes)
        ]

    def set_data(self, X, y):
        for i in range(self.classes):
            y_train = (y == i).astype(int)
            self.classifiers[i].set_data(X, y_train)

    def predict(self, X):
        h = []
        for cls in self.classifiers:
            h.append(cls.calculate_hypothesis(X))

        return np.argmax(np.array(h), axis=0)

    def accuracy(self, X, y):
        errors = self.predict(X) - y
        return 1.0 - ((float(np.count_nonzero(errors))) / len(errors))

mlr = MulticlassLogisticRegression()
mlr.set_data(X, y)

pred_value = mlr.predict(X[-1])
print(pred_value, y[-1])
print(mlr.accuracy(X, y))
