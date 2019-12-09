import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

df = load_boston()
X = df['data'][:, :-1]
y = df['data'][:, -1]
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
print(f'X_train, y_train shape: {X_train.shape}, {y_train.shape}')
print(f'X_test, y_test shape: {X_test.shape}, {y_test.shape}')


class GradientBoostingRegressor:
    def __init__(self, etha=0.05, trees_count=50, **tree_params):
        self.ethas = np.array([etha(i) if callable(etha) else etha for i in range(trees_count)])
        self.tree_params = tree_params
        self.trees_count = trees_count
        self.trees = []

    def _negative_gradient(self, predictions, y):
        return y - predictions

    def predict(self, X, trees_limit=None):
        trees = self.trees[:trees_limit] if trees_limit else self.trees
        predictions = np.array([tree.predict(X) for tree in trees]).T
        return np.dot(predictions, self.ethas[:len(trees)])

    def fit(self, X, Y):
        for i in range(self.trees_count):
            predictions = self.predict(X)
            gradients = self._negative_gradient(predictions, Y)

            decision_tree = DecisionTreeRegressor(**self.tree_params)
            decision_tree.fit(X, gradients)
            self.trees.append(decision_tree)

    def score(self, X, Y, error_function=mean_squared_error, **predict_params):
        return error_function(Y, self.predict(X, **predict_params))


trees_count = 50
etha = 0.9

regressor = GradientBoostingRegressor(etha, max_depth=5, random_state=42)
regressor.fit(X_train, y_train)
print('Prediction params:\nEthas: 0.9\n')
print('Prediction error on train set: ', regressor.score(X_train, y_train))
print('Prediction error on test set: ', regressor.score(X_test, y_test))

etha = lambda i: 0.9 / (1.0 + i)

regressor = GradientBoostingRegressor(etha, max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

print('Prediction params:\nEthas: 0.9 / (1.0 + i)\n')
print('Prediction error on train set: ', regressor.score(X_train, y_train))
print('Prediction error on test set: ', regressor.score(X_test, y_test))

trees_count = np.arange(1, 50, 2)
max_count = trees_count.max()

regressor = GradientBoostingRegressor(etha, trees_count=max_count, max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

Y_train = np.array([regressor.score(X_train, y_train, trees_limit=count) for count in trees_count])
Y_test = np.array([regressor.score(X_test, y_test, trees_limit=count) for count in trees_count])

best_trees_count = trees_count[np.argmin(Y_test)]

plt.plot(trees_count, Y_train, marker='.', label="train")
plt.plot(trees_count, Y_test, marker='.', label="test")
plt.xlabel('Iteration count (trees count)')
plt.ylabel('Error')
plt.legend()
plt.show()

trees_depth = np.arange(1, 20)
Y_train, Y_test = [], []

for depth in trees_depth:
    regressor = GradientBoostingRegressor(etha, max_depth=depth, random_state=42)
    regressor.fit(X_train, y_train)
    Y_train.append(regressor.score(X_train, y_train))
    Y_test.append(regressor.score(X_test, y_test))

best_trees_depth = trees_depth[np.argmin(np.array(Y_test))]

plt.plot(trees_depth, Y_train, marker='.', label="train")
plt.plot(trees_depth, Y_test, marker='.', label="test")
plt.xlabel('Trees depth')
plt.ylabel('Error')
plt.legend()
plt.show()

regressor = GradientBoostingRegressor(etha, trees_count=best_trees_count,
                                      max_depth=best_trees_depth, random_state=42)
regressor.fit(X_train, y_train)
rmse = np.sqrt(regressor.score(X_test, y_test))
print('RMSE on Gradient Boosting regression: ', rmse)

regressor = LinearRegression().fit(X_train, y_train)
predictions = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print('RMSE on Linear regression: ', rmse)
