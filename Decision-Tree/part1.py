import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy


class Node:
    def __init__(self, feature_name=None, info_gain=-1, depth=-1):
        self.feature_name = feature_name
        self.info_gain = info_gain
        self.left = None
        self.right = None
        self.leaf = None
        self.depth = depth
        self.res = None


def info_entropy(df):
    p = np.sum(df['label'] == 1)
    n = np.sum(df['label'] == 0)

    sample_num = df.shape[0]
    entropy = 0
    if p == 0 or n == 0:
        return entropy
    entropy = -((p / sample_num) * np.log2(p / sample_num) + (n / sample_num) * np.log2(n / sample_num))
    return entropy


def info_gain(df, feat):
    left_sub = df[df[feat] == 0]
    right_sub = df[df[feat] == 1]
    conditional_entropy = (left_sub.shape[0] / df.shape[0]) * info_entropy(left_sub) + (
            right_sub.shape[0] / df.shape[0]) * info_entropy(right_sub)
    gain = info_entropy(df) - conditional_entropy
    return gain


def get_best_feature_name_by_info(df, features_name):
    max_info_gain = -np.inf
    best_feature = None
    for feat in features_name:
        gain = info_gain(df, feat)
        if gain > max_info_gain:
            max_info_gain = gain
            best_feature = feat
    return best_feature, max_info_gain


def build_decision_tree(train_data, features_name, depth=1, dmax=2):
    p = np.sum(train_data['label'] == 1)
    n = np.sum(train_data['label'] == 0)
    # leaf node or touch dmax
    if p == train_data.shape[0] or n == train_data.shape[0] or depth >= dmax or len(features_name) == 0:
        leaf = Node()
        if depth < dmax:
            leaf.leaf = True
        leaf.depth = depth
        if p > n:
            leaf.res = 1
        else:
            leaf.res = 0
        return leaf
    best_feat, max_info_gain = get_best_feature_name_by_info(train_data, features_name)
    new_features_name = copy.deepcopy(features_name)
    new_features_name.remove(best_feat)
    tree = Node(best_feat, max_info_gain, depth)
    left_sub = train_data[train_data[best_feat] == 0]
    right_sub = train_data[train_data[best_feat] == 1]
    # Recursively build left and right subtree
    tree.left = build_decision_tree(left_sub, new_features_name, depth=depth + 1, dmax=dmax)
    tree.right = build_decision_tree(right_sub, new_features_name, depth=depth + 1, dmax=dmax)
    return tree


def predictions(root, df):
    pred = []

    def predict(root, df):
        if root.res is not None:
            return root.res
        if df[root.feature_name] == 0:
            return predict(root.left, df)
        elif df[root.feature_name] == 1:
            return predict(root.right, df)

    for index, row in df.iterrows():
        pred.append(predict(root, row))
    return np.array(pred)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


def split_print(tree: Node, d=3):
    from collections import deque
    q = deque()
    q.append(tree)
    while len(q) > 0:
        top = q.popleft()
        if top.depth > d:
            return
        print(top.feature_name, top.info_gain)
        if top.left is not None:
            q.append(top.left)
        if top.right is not None:
            q.append(top.right)


if __name__ == '__main__':
    # load trainset
    X_train = pd.read_csv('pa4_train_X.csv')
    y_train = pd.read_csv('pa4_train_y.csv', header=None)
    y_train.columns = ['label']
    # load devset
    X_dev = pd.read_csv('pa4_dev_X.csv')
    y_dev = pd.read_csv('pa4_dev_y.csv', header=None)
    y_dev = y_dev.to_numpy().reshape(y_dev.shape[0])

    train_data = pd.concat((X_train, y_train), axis=1)
    features_name = train_data.columns.tolist()
    features_name.remove('label')

    y_train = y_train.to_numpy().reshape(y_train.shape[0], )
    split = True
    # d_max = [2, 5, 10, 20, 25, 30, 40, 50]
    d_max = [2, 5, 10, 20, 25, 30, 40, 50]
    acc_train = []
    acc_dev = []
    import tqdm

    for d in tqdm.tqdm(d_max):
        model = build_decision_tree(train_data, features_name, dmax=d)
        if print:
            split_print(model)
        acc_train.append(accuracy(predictions(model, X_train), y_train))
        acc_dev.append(accuracy(predictions(model, X_dev), y_dev))
    # plot
    plt.xlabel('dmax')
    plt.ylabel('accuracy')
    plt.plot(d_max, acc_train, label='acc_train')
    plt.plot(d_max, acc_dev, label='acc_dev')
    plt.legend()
    plt.show()
