import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

FILENAME = './data/ex3data1.mat'


def reshape(vector):
    return np.reshape(vector, vector.shape[0])


mat = loadmat(FILENAME)
X_train, y_train = mat['X'], reshape(mat['y'])
X_val, y_val = mat['Xval'], reshape(mat['yval'])
X_test, y_test = mat['Xtest'], reshape(mat['ytest'])


# fig, ax = plt.subplots()
# ax.plot(X_train, y_train, 'o')
# ax.set_xlabel('water level change')
# ax.set_ylabel('water volume')
# plt.show()


def normalize(X):
    normalized_X = X.copy()
    for i in range(X.shape[1]):
        feature = X[:, i]
        delta = np.max(feature) - np.min(feature)
        normalized_X[:, i] = (feature - np.mean(feature)) / delta
    return normalized_X


class LinearRegressionRegularized:
    THRESHOLD = 1e-6

    def __init__(self, learning_rate=0.001, max_steps=100000, normalized=False, regularized_param=0, degree=1):
        self.weights = []
        self.costs = []
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.regularized_param = regularized_param
        self.normalized = normalized
        self.degree = degree

    def set_data(self, X, y):
        X = X.astype('float64')
        y = y.astype('float64')
        if self.degree > 1:
            X = self.build_poly_features(X)
        if self.normalized:
            X = normalize(X)
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.weights = np.zeros(X.shape[1])
        self.gradient_descent(X, y)

    def calculate_hypothesis(self, X):
        if len(X.shape) > 1 and X.shape[1] < self.weights.shape[0]:
            X = np.column_stack((np.ones(X.shape[0]), X))

        return X.dot(self.weights)

    def cost_function(self, X, y):
        predictions = self.calculate_hypothesis(X)
        cost = np.mean((predictions - y) ** 2) / 2
        regularized_weights = self.weights[1:]
        regularized_cost = self.regularized_param / 2 / X.shape[0] * np.dot(regularized_weights.T, regularized_weights)
        return cost + regularized_cost

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

    def calculate_derivative(self, X, y):
        predictions = self.calculate_hypothesis(X)
        diff = predictions - y
        new_weights = np.dot(X.T, diff) / X.shape[0]
        new_weights_regularized = (self.regularized_param / X.shape[0]) * new_weights[1:]
        new_weights_regularized = np.insert(new_weights_regularized, 0, 0)
        return new_weights + new_weights_regularized

    def gradient_descent_step(self, X, y):
        delta = self.calculate_derivative(X, y)
        delta *= self.learning_rate
        self.weights -= delta

    def predict(self, X):
        X = np.insert(X, 0, 1)
        return self.calculate_hypothesis(X)

    def build_poly_features(self, X):
        x = X.reshape(X.shape[0])
        res = np.array(x)

        for i in range(2, self.degree + 1):
            res = np.column_stack((res, x ** i))

        return res


# lr = LinearRegressionRegularized()
# lr.set_data(X_train, y_train)
#
# min_x, max_x = int(min(X_train)), int(max(X_train)) + 1
# xi = list(range(min_x, max_x + 1))
# line = [lr.predict(np.array(i)) for i in xi]
# plt.plot(X_train, y_train, 'o', xi, line)
# plt.xlabel('water level change')
# plt.ylabel('water volume')
# plt.show()

def learning_curves(lr, X_train, y_train, X_val, y_val, X_train_poly=None):
    z_train = np.zeros(X_train.shape[0])
    z_val = np.zeros(X_train.shape[0])
    for i in range(1, X_train.shape[0]):
        lr.set_data(X_train[:i + 1], y_train[:i + 1])
        z_train[i] = lr.cost_function(X_train if X_train_poly is None else X_train_poly, y_train)
        z_val[i] = lr.cost_function(X_val, y_val)

    plt.plot(range(1, X_train.shape[0]), z_train[1:], c="r")
    plt.plot(range(1, X_train.shape[0]), z_val[1:], c="b")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.legend(["Training", "Validation"], loc="best")
    plt.show()


# lr = LinearRegressionRegularized()
# learning_curves(lr, X_train, y_train, X_val, y_val)

def plot_train_and_fit(model, X, y):
    x = np.linspace(min(X), max(X), 1000)
    X_poly = model.build_poly_features(x)
    X_norm = normalize(X_poly)
    line = [model.predict(i) - 1.5 for i in X_norm]
    plt.plot(X, y, 'o', x, line, markersize=4)
    plt.xlabel('water level change')
    plt.ylabel('water volume')
    plt.show()

# lr = LinearRegressionRegularized(learning_rate=1, normalized=True, degree=8)
# lr.set_data(X_train, y_train)
# plot_train_and_fit(lr, X_train, y_train)
# X_train_poly, X_val_poly = lr.build_poly_features(X_train), lr.build_poly_features(X_val)
# X_train_norm, X_val_norm = normalize(X_train_poly), normalize(X_val_poly)
# learning_curves(lr, X_train, y_train, X_val_norm, y_val, X_train_norm)


# lr = LinearRegressionRegularized(learning_rate=1, normalized=True, degree=8, regularized_param=1)
# lr.set_data(X_train, y_train)
# plot_train_and_fit(lr, X_train, y_train)
# X_train_poly, X_val_poly = lr.build_poly_features(X_train), lr.build_poly_features(X_val)
# X_train_norm, X_val_norm = normalize(X_train_poly), normalize(X_val_poly)
# learning_curves(lr, X_train, y_train, X_val_norm, y_val, X_train_norm)

#
# lr = LinearRegressionRegularized(learning_rate=0.01, normalized=True, degree=8, regularized_param=100)
# lr.set_data(X_train, y_train)
# plot_train_and_fit(lr, X_train, y_train)
# X_train_poly, X_val_poly = lr.build_poly_features(X_train), lr.build_poly_features(X_val)
# X_train_norm, X_val_norm = normalize(X_train_poly), normalize(X_val_poly)
# learning_curves(lr, X_train, y_train, X_val_norm, y_val, X_train_norm)


regularized_values = [0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.1, 0.3, 1]
validation_errors = []

for regularized_param in regularized_values:
    lr = LinearRegressionRegularized(learning_rate=0.5, normalized=True, regularized_param=regularized_param, degree=8)
    lr.set_data(X_train, y_train)
    cost = lr.cost_function(lr.build_poly_features(X_val), y_val)
    validation_errors.append(cost)

OPTIMAL_LAMBDA = regularized_values[np.argmin(np.array(validation_errors))]
print(f"Minimum error is for lambda = {OPTIMAL_LAMBDA}")
plt.plot(list(range(len(regularized_values))), validation_errors)
# plt.axis([0, len(regularized_values), 0, max(regularized_values) + 1])
plt.xlabel("lambda")
plt.ylabel("error")
plt.title("Validation Curve")
plt.grid()
plt.show()

lr_test = LinearRegressionRegularized(learning_rate=1, normalized=True, regularized_param=OPTIMAL_LAMBDA, degree=8)
lr_test.set_data(X_train, y_train)

X_test_poly = normalize(lr_test.build_poly_features(X_test))
final_cost = lr_test.cost_function(X_test_poly, y_test)
print(f"Error on test set is {final_cost}")
