import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

path = 'data/'


def puf(max_attrsize=128, max_files=3, max_lines=1000):
    import os

    files = [path + file for file in os.listdir(path)][:max_files]
    x = []
    y = []
    for file in files:
        with open(file) as f:
            contents = f.readlines()[:max_lines]
            for content in contents:
                sub_x, sub_y = content.split()
                x_elem = [0 for _ in range(max_attrsize - len(sub_x))] + [int(ch) for ch in sub_x]
                x.append(x_elem)
                y.append(int(sub_y))
    x = np.array(x)
    y = np.array(y)
    return x, y


x, y = puf(max_lines=10000)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.75, test_size=.25, random_state=42)
print(f'X shape: {x.shape}')
print(f'Y shape: {y.shape}')


def fit_predict_sample():
    regressor = DecisionTreeRegressor()
    regressor.fit(x_train, y_train)
    a1_pred = regressor.predict(x_test)
    sample = ''.join(map(str, x_test[0]))
    print("Prediction sample x = {}, y = {}".format(sample, int(a1_pred[0])))


fit_predict_sample()

learn_algos = [DecisionTreeRegressor, LinearRegression, GradientBoostingRegressor]


def learn(x_train, y_train, x_test, alg=DecisionTreeRegressor):
    regressor = alg()
    regressor.fit(x_train, y_train)
    return regressor.predict(x_test)


def regressors(learn_algos):
    preds = [learn(x_train, y_train, x_test, alg=alg) for alg in learn_algos]

    accuracy_algs = [sklearn.metrics.mean_squared_error,
                     sklearn.metrics.mean_absolute_error,
                     sklearn.metrics.median_absolute_error]

    accs = [alg(y_test, preds[0]) for alg in accuracy_algs]
    print("metrics")
    print(accs)


regressors(learn_algos)

from sklearn.svm import LinearSVC

learn_algos = [DecisionTreeClassifier, LinearSVC, GradientBoostingClassifier]
alg_names = ['DecisionTreeClassifier', 'LinearSVC', 'GradientBoostingClassifier']


def best_classifier(learn_algos, alg_names):
    preds = [learn(x_train, y_train, x_test, alg=alg) for alg in learn_algos]
    accs = [sklearn.metrics.accuracy_score(y_test, pred) for pred in preds]

    print("Classifiers")
    for i, alg in enumerate(learn_algos):
        print(f'{alg_names[i]} - {accs[i]}')
    print("Best accuracy: {}".format(alg_names[accs.index(max(accs))]))


best_classifier(learn_algos, alg_names)


def get_accuracy(x_train, x_test, y_train, y_test):
    pred = learn(x_train, y_train, x_test, alg=GradientBoostingClassifier)
    return sklearn.metrics.accuracy_score(y_test, pred)


gen_x = []
gen_y = []

for ml in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
    gen_x.append(ml)
    x, y = puf(max_lines=ml, max_files=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.85, test_size=.15, random_state=42)
    gen_y.append(get_accuracy(x_train, x_test, y_train, y_test))

x_train_lin = np.array([[i] for i in gen_y])
y_train_lin = np.array(gen_x)

lin = LinearRegression()
lin.fit(x_train_lin, y_train_lin)
predicted_training_set_count = lin.predict(np.array([0.95]).reshape(-1, 1))[0]
print(f'Предположительный размер выборки: {int(predicted_training_set_count)}')

print(f'N list - {gen_x}')
print(f'Accuracy list - {gen_y}')
plt.plot(gen_x, gen_y, "go")

plt.show()
