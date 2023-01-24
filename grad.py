# %%
from operator import index
from turtle import color
import numpy as np
import pandas as pd
import prince
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


n_feature = 5
n_class = 2

# %%


def make_network(n_hidden=100):
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )

    return model


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h @ model['W2'])

    return h, prob


def backward(model, xs, hs, errs):
    dW2 = hs.T @ errs

    dh = errs @ model['W2'].T
    dh[hs < 0] = 0
    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)

# %%

def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.
        err = y_true - y_pred

        xs.append(x)
        hs.append(h)
        errs.append(err)

    return backward(model, np.array(xs), np.array(hs), np.array(errs))


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


def sgd(model, X_train, y_train, minibatch_size):
    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            model[layer] += alpha * grad[layer]

    return model


def momentum(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model


def nesterov(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        model_ahead = {k: v + gamma * velocity[k] for k, v in model.items()}
        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model


def adagrad(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            cache[k] += grad[k]**2
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model


def rmsprop(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            cache[k] = gamma * cache[k] + (1 - gamma) * (grad[k]**2)
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model


def adam(model, X_train, y_train, minibatch_size):
    M = {k: np.zeros_like(v) for k, v in model.items()}
    R = {k: np.zeros_like(v) for k, v in model.items()}
    beta1 = .9
    beta2 = .999

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        t = iter
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            M[k] = beta1 * M[k] + (1. - beta1) * grad[k]
            R[k] = beta2 * R[k] + (1. - beta2) * grad[k]**2

            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            model[k] += alpha * m_k_hat / (np.sqrt(r_k_hat) + eps)

    return model


def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]

# %%
X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

n_iter = 100
eps = 1e-8  # Smoothing to avoid division by zero
alpha = 1e-2
minibatch_size = 50
n_experiment = 3

algos = dict(
    sgd=sgd,
    momentum=momentum,
    adagrad=adagrad,
    adam=adam
)

algo_accs = {k: np.zeros(n_experiment) for k in algos}

for algo_name, algo in algos.items():
    print('Experimenting on {}'.format(algo_name))

    for k in range(n_experiment):
        # print('Experiment-{}'.format(k))

        # Reset model
        model = make_network()
        model = algo(model, X_train, y_train, minibatch_size)

        y_pred = np.zeros_like(y_test)

        for i, x in enumerate(X_test):
            _, prob = forward(x, model)
            y = np.argmax(prob)
            y_pred[i] = y

        algo_accs[algo_name][k] = np.mean(y_pred == y_test)

print()

for k, v in algo_accs.items():
    print('{} => mean accuracy: {}, std: {}'.format(k, v.mean(), v.std()))

# %%

# DATA
from prince import FAMD, MCA, PCA
from enum import Enum
emp_df = pd.read_csv('data/HR Employee Attrition.csv')

"""
Education
1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'

EnvironmentSatisfaction
1 'Low' 2 'Medium' 3 'High' 4 'Very High'

JobInvolvement
1 'Low' 2 'Medium' 3 'High' 4 'Very High'

JobSatisfaction
1 'Low' 2 'Medium' 3 'High' 4 'Very High'

PerformanceRating
1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'

RelationshipSatisfaction
1 'Low' 2 'Medium' 3 'High' 4 'Very High'

WorkLifeBalance
1 'Bad' 2 'Good' 3 'Better' 4 'Best'
"""
emp_df.head()

class DimensionalityReductionMethod(str, Enum):
    """Dimensionality reduction method enum"""

    PCA = "PCA"
    MCA = "MCA"
    FAMD = "FAMD"
# %%
features = emp_df.copy()
features

le = LabelEncoder()
y_og = pd.Series(le.fit_transform(features.Attrition))

# label_summary = pd.DataFrame(zip(y, features.Attrition), columns = ['label_encode', 'Attrition']).groupby(['label_encode', 'Attrition']).size().to_frame('count').reset_index()
# label_summary['%'] = round(label_summary['count']/label_summary['count'].sum()*100, 1)
# label_summary


# X_train, X_test, y_train, y_test = train_test_split(features.drop('Attrition', axis = 1), y, test_size = 0.2,
                                                    # random_state = 778, shuffle = True, stratify = y)

# # check distribution
# pd.DataFrame({'Count - Train': y_train.value_counts(), '% - Train': round(y_train.value_counts(1)*100, 1),
#               'Count - Test': y_test.value_counts(), '% - Test': round(y_test.value_counts(1)*100, 1)})

# %%
# Encoding
from sklearn.compose import ColumnTransformer
nominal = X_train.select_dtypes(include=['object']).columns
ohe = ColumnTransformer([('encoder', OneHotEncoder(), nominal)], remainder='passthrough')
ohe
ohe.fit(X_train)
X_train_new = ohe.transform(X_train)
X_test_new = ohe.transform(X_test)
# %%
categorical_var = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
       'MaritalStatus', 'Over18', 'OverTime']

features_ohe = pd.get_dummies(features, columns= categorical_var)
features_ohe

# %%
features_pca = features.drop(columns=["Attrition"]).copy()
X = features_ohe.drop(columns=["Attrition"]).copy()
# %%

dim_red = {
            DimensionalityReductionMethod.PCA: PCA,
            DimensionalityReductionMethod.MCA: MCA,
            DimensionalityReductionMethod.FAMD: FAMD,
        }["PCA"]
dim_red

scaled_features_pca = (
            dim_red(n_components = n_feature)
            .fit_transform(X)
            .values
        )

dim_red = {
            DimensionalityReductionMethod.PCA: PCA,
            DimensionalityReductionMethod.MCA: MCA,
            DimensionalityReductionMethod.FAMD: FAMD,
        }["FAMD"]
dim_red

scaled_features_famd = (
            dim_red(n_components = n_feature)
            .fit_transform(features_pca)
            .values
        )

# %%

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

pd.options.display.max_columns = None
%matplotlib inline

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_features_pca, y_og, random_state=42)

n_iter = 100
eps = 1e-8  # Smoothing to avoid division by zero
alpha = 1e-2
minibatch_size = 100
n_experiment = 10

algos = dict(
    sgd=sgd,
    momentum=momentum,
    adagrad=adagrad,
    adam=adam
)

algo_accs_pca = {k: np.zeros(n_experiment) for k in algos}
print("WITH PCA")
for algo_name, algo in algos.items():
    print('Experimenting on {}'.format(algo_name))

    for k in range(n_experiment):
        # print('Experiment-{}'.format(k))

        # Reset model
        model = make_network()
        model = algo(model, X_train, y_train, minibatch_size)

        y_pred = np.zeros_like(y_test)

        for i, x in enumerate(X_test):
            _, prob = forward(x, model)
            y = np.argmax(prob)
            y_pred[i] = y

        algo_accs_pca[algo_name][k] = np.mean(y_pred == y_test)

print()

for k, v in algo_accs_pca.items():
    print('{} => mean accuracy: {}, std: {}'.format(k, v.mean(), v.std()))

# %%


# %%

X_train, X_test, y_train, y_test = train_test_split(scaled_features_famd, y_og, random_state=42)

n_iter = 100
eps = 1e-8  # Smoothing to avoid division by zero
alpha = 1e-2
minibatch_size = 100
n_experiment = 10

algos = dict(
    sgd=sgd,
    momentum=momentum,
    adagrad=adagrad,
    adam=adam
)

algo_accs_famd = {k: np.zeros(n_experiment) for k in algos}
print("WITH FAMD")
for algo_name, algo in algos.items():
    print('Experimenting on {}'.format(algo_name))

    for k in range(n_experiment):
        # print('Experiment-{}'.format(k))

        # Reset model
        model = make_network()
        model = algo(model, X_train, y_train, minibatch_size)

        y_pred = np.zeros_like(y_test)

        for i, x in enumerate(X_test):
            _, prob = forward(x, model)
            y = np.argmax(prob)
            y_pred[i] = y

        algo_accs_famd[algo_name][k] = np.mean(y_pred == y_test)
        # print(classification_report(y_test, y_pred))
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap = 'Greens')

print()

for k, v in algo_accs_famd.items():
    print('{} => mean accuracy: {}, std: {}'.format(k, v.mean(), v.std()))
# %%

import torch
import torch.nn.functional as F
from torchcontrib.optim.swa import SWA

for k in range(n_experiment):
        # print('Experiment-{}'.format(k))

        # Reset model
        model_swa = make_network()
        model_swa = SWA(model_swa, X_train, y_train, minibatch_size)

        y_pred = np.zeros_like(y_test)

        for i, x in enumerate(X_test):
            _, prob = forward(x, model_swa)
            y = np.argmax(prob)
            y_pred[i] = y

        algo_accs_famd["swa"][k] = np.mean(y_pred == y_test)

        # print(classification_report(y_test, y_pred))
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap = 'Greens')




# %%
import seaborn as sns
import matplotlib.pyplot as plt
# GRAPHS REPRESENTATIONS!
r1 = pd.DataFrame(algo_accs_pca)
r2 = pd.DataFrame(algo_accs_famd)

sns.lineplot(data=r1[["sgd","momentum", "adam"]])

# %%

r2["iteration"] = r2.index+1
r2
r1["iteration"] = r1.index+1
r1
r1
r2

for col in ["sgd","momentum","adagrad","adam"]:
    sns.lineplot(x=r1.iteration, y=r1[col], color="blue", label="PCA",linestyle="-")
    sns.lineplot(x=r2.iteration,y= r2[col], color="green", label="FAMD",linestyle="-")
    plt.show()
    plt.clf()

# %%

import numpy as np 
import matplotlib.pyplot as plt 
from importlib import reload
plt=reload(plt)
  
# r1_means = []
# for k, v in r1.items():
#     r1_means.append(v.mean())
# r1_means

X = r1.columns.to_list()
r1_mean = [v.mean() for k,v in r1.items() ]
r2_mean = [v.mean() for k,v in r2.items() ]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, r1_mean, 0.4, color='blue', edgecolor = 'black',label = 'PCA')
plt.bar(X_axis + 0.2, r2_mean, 0.4, color='green', edgecolor = 'black',label = 'FAMD')
  
plt.xticks(X_axis, X)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Mean Accuracy of algorithms with PCA and FAMD")
#plt.legend()

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

#sns.barplot(data = r1)