import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import minmax_scale


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def loss_func(y, pred):
    return (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).sum()


def grad_func(X, y, pred):
    return X.T @ (pred - y) / y.size


def accuracy_score(X, y, w):
    pred = sigmoid(np.dot(X, w))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    acc = np.sum(pred == y) / y.size
    return acc


def logistic_L1(X, y, alpha=1e-1, lambda_cofee=1e-2, max_iter=200, eps=1e-4):
    # data num
    N = X.shape[0]
    # feature num
    d = X.shape[1]
    # init w
    w = np.ones(X.shape[1])
    # history
    acc_his = []
    loss_his = []
    grad_his = []
    for i in range(int(max_iter)):
        acc = accuracy_score(X, y, w)
        pred = sigmoid(np.dot(X, w))
        loss = loss_func(y, pred)
        grad = grad_func(X, y, pred)
        # convergence?
        if np.linalg.norm(grad) < eps:
            break
        # recored history
        acc_his.append(acc)
        loss_his.append(loss)
        grad_his.append(np.linalg.norm(grad))

        # update w
        w = w + alpha / N * ((y - (sigmoid(X @ w))) @ X)
        # L1
        w = np.sign(w) * np.maximum(np.abs(w) - alpha * lambda_cofee, 0)

    fig, ax = plt.subplots(3, figsize=(13, 13))
    ax[0].set_title('Accurary')
    ax[0].plot(acc_his)
    ax[1].set_title('Loss')
    ax[1].plot(loss_his)
    ax[2].set_title('GradNorm')
    ax[2].plot(grad_his)
    plt.show()
    return w


if __name__ == '__main__':
    # Load
    X_train = pd.read_csv('pa2_train_X.csv')
    y_train = pd.read_csv('pa2_train_y.csv')
    X_val = pd.read_csv('pa2_dev_X.csv')
    y_val = pd.read_csv('pa2_dev_y.csv')
    features = X_train.columns

    # Preprocessing
    X_train[['Age', 'Annual_Premium', 'Vintage']] = minmax_scale(X_train[['Age', 'Annual_Premium', 'Vintage']])
    X_val[['Age', 'Annual_Premium', 'Vintage']] = minmax_scale(X_val[['Age', 'Annual_Premium', 'Vintage']])
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()

    # different lambda
    w0 = logistic_L1(X_train, y_train, lambda_cofee=0)
    w1 = logistic_L1(X_train, y_train, lambda_cofee=1e-1)
    w2 = logistic_L1(X_train, y_train, lambda_cofee=1e-2)
    w3 = logistic_L1(X_train, y_train, lambda_cofee=1e-3)
    w4 = logistic_L1(X_train, y_train, lambda_cofee=1e-4)
    w5 = logistic_L1(X_train, y_train, lambda_cofee=1e-5)

    # metirc
    w0_train_acc = accuracy_score(X_train, y_train, w0)
    w0_val_acc = accuracy_score(X_val, y_val, w0)
    w1_train_acc = accuracy_score(X_train, y_train, w1)
    w1_val_acc = accuracy_score(X_val, y_val, w1)
    w2_train_acc = accuracy_score(X_train, y_train, w2)
    w2_val_acc = accuracy_score(X_val, y_val, w2)
    w3_train_acc = accuracy_score(X_train, y_train, w3)
    w3_val_acc = accuracy_score(X_val, y_val, w3)
    w4_train_acc = accuracy_score(X_train, y_train, w4)
    w4_val_acc = accuracy_score(X_val, y_val, w4)
    w5_train_acc = accuracy_score(X_train, y_train, w5)
    w5_val_acc = accuracy_score(X_val, y_val, w5)

    train_acc = [w0_train_acc, w1_train_acc, w2_train_acc, w3_train_acc, w4_train_acc, w5_train_acc]
    val_acc = [w0_val_acc, w1_val_acc, w2_val_acc, w3_val_acc, w4_val_acc, w5_val_acc]
    labels = ['1e0', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5']
    w_list = [w0, w1, w2, w3, w4, w5]

    # plot
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.bar(x - width / 2, train_acc, width, label='TrainSetAcc')
    ax.bar(x + width / 2, val_acc, width, label='ValSetAcc')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.75, 0.8)
    ax.set_title('Accuracy of Train and Val with L2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

    # sorted features
    sorted_features = features[np.argsort(np.abs(w_list[np.argmax(val_acc)]))[::-1]]
    # top 5
    print(sorted_features[:5])
    # zero num
    print(np.sum(np.abs(w_list[np.argmax(val_acc)]) < 1e-6))
