import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy


class Node:
    def __init__(self, feature_name=None, info_gain=-1):
        self.feature_name = feature_name
        self.info_gain = info_gain
        self.left = None
        self.right = None
        self.leaf = None
        self.depth = None
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


def build_decision_tree(train_data, features_name, m, depth=1, dmax=2):
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
    # sample features
    sub_sample_features_name = list(np.array(features_name)[np.random.permutation(len(range(len(features_name))))[:m]])
    # get best feature
    best_feat, max_info_gain = get_best_feature_name_by_info(train_data, sub_sample_features_name)
    new_features_name = copy.deepcopy(features_name)
    new_features_name.remove(best_feat)  # remove used feature for next building

    tree = Node(best_feat, max_info_gain)
    left_sub = train_data[train_data[best_feat] == 0]
    right_sub = train_data[train_data[best_feat] == 1]
    # Recursively build left and right subtree
    tree.left = build_decision_tree(left_sub, new_features_name, m, depth=depth + 1, dmax=dmax)
    tree.right = build_decision_tree(right_sub, new_features_name, m, depth=depth + 1, dmax=dmax)
    return tree


def predictions(sub_tree_cls, df):
    res = []

    def predict(root, df):
        if root.res is not None:
            return root.res
        if df[root.feature_name] == 0:
            return predict(root.left, df)
        elif df[root.feature_name] == 1:
            return predict(root.right, df)

    for index, row in df.iterrows():
        pred_list = []
        for tree in sub_tree_cls:
            pred_list.append(predict(tree, row))
        # get counts, counts max one as res
        pred_label_counts = collections.Counter(pred_list)
        res.append(max(zip(pred_label_counts.values(), pred_label_counts.keys()))[1])
    return np.array(res)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


if __name__ == '__main__':
    # load trainset
    X_train = pd.read_csv('pa4_train_X.csv')
    y_train = pd.read_csv('pa4_train_y.csv', header=None)
    y_train.columns = ['label']
    # load devset
    X_dev = pd.read_csv('pa4_dev_X.csv')
    y_dev = pd.read_csv('pa4_dev_y.csv', header=None)
    # get train_data
    train_data = pd.concat((X_train, y_train), axis=1)
    features_name = train_data.columns.tolist()
    features_name.remove('label')

    # get label
    y_dev = y_dev.to_numpy().reshape(y_dev.shape[0])
    y_train = y_train.to_numpy().reshape(y_train.shape[0], )
    d_max = [2, 10, 25]
    m_ = [5, 25, 50, 100]
    T_ = [i for i in range(10, 110, 10)]
    np.random.seed(1)
    import tqdm

    for d in tqdm.tqdm(d_max, desc='dmax'):
        # init plot
        fig, ax = plt.subplots(2, figsize=(12, 12))
        fig.suptitle('dmax={}'.format(d), fontsize=16)
        ax[0].set_xlabel('T')
        ax[0].set_ylabel('Accuracy')
        ax[1].set_xlabel('T')
        ax[1].set_ylabel('Accuracy')
        ax[0].set_title('Train_acc')
        ax[1].set_title('Val_acc')
        for m in tqdm.tqdm(m_, desc='m'):
            tree_list = []
            # build forest(100 T)
            for i in tqdm.trange(100):
                sample_train_data = train_data.iloc[np.random.randint(0, train_data.shape[0], train_data.shape[0])]
                model = build_decision_tree(sample_train_data, features_name, m, dmax=d)
                tree_list.append(model)
            tree_list = np.array(tree_list)
            acc_train = []
            acc_dev = []
            for T in T_:
                # sample T trees and accuracy on trainset and valset
                sub_tree_cls = tree_list[np.random.permutation(100)[:T]]
                acc_train.append(accuracy(y_train, predictions(sub_tree_cls, X_train)))
                acc_dev.append(accuracy(y_dev, predictions(sub_tree_cls, X_dev)))
            ax[0].plot(T_, acc_train, label='m={}'.format(m))
            ax[1].plot(T_, acc_dev, label='m={}'.format(m))

        ax[0].legend()
        ax[1].legend()
        plt.show()
