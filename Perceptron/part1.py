#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use("TKAgg")

#read raw data
X_train = pd.read_csv("./pa3_data/pa3_train_X.csv")
y_train = pd.read_csv("./pa3_data/pa3_train_y.csv")
X_dev = pd.read_csv("./pa3_data/pa3_dev_X.csv")
y_dev = pd.read_csv("./pa3_data/pa3_dev_y.csv")

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_dev shape: ", X_dev.shape)
print("y_dev shape: ", y_dev.shape)

X_train_np = np.array(X_train)
y_train_np = np.squeeze(np.array(y_train))
X_dev_np = np.array(X_dev)
y_dev_np = np.squeeze(np.array(y_dev))

def avg_per(X, y, maxiter = 100):
    num_instances, num_feats = X.shape
    w = np.zeros(num_feats)
    w_bar = np.zeros(num_feats)
    s = 1
    while s < maxiter:
        for i in range(num_instances):
            xi = X[i]
            yi = y[i]
            if yi* ((w.T) @ xi) <= 0:
                w = w + yi * xi
            w_bar = (s*w_bar + w)/(s+1)
            s = s + 1
    return w, w_bar


def pre(w, X):
    ans =  np.matmul(X, w)
    return np.sign(ans)

def acc(gt_y, pre_y):
    num = pre_y.shape
    pre_correct = np.sum((pre_y == gt_y)*1)
    return np.squeeze(pre_correct/num)

def avg_per_acc(X_train, y_train, X_dev, y_dev, maxiter = 100):
    acc_records = []
    num_instances, num_feats = X_train.shape
    w = np.zeros(num_feats)
    w_bar = np.zeros(num_feats)
    s = 1
    t = 0
    while t < maxiter:
        for i in range(num_instances):
            xi = X_train[i]
            yi = y_train[i]
            h = np.dot(xi,w) *yi
            if h <= 0:
                w = w + xi * yi
            w_bar = (s*w_bar + w)/(s+1)
            s = s + 1
        t = t+1
        if t%10 ==0:
            print("iter {}".format(t))
            print(t, acc_train_w, acc_dev_w, acc_train_w_bar,acc_dev_w_bar)
        y_train = np.squeeze(y_train)
        y_dev = np.squeeze(y_dev)
        # for w
        pre_train_w = pre(w, X_train)
        pre_dev_w = pre(w, X_dev)
        acc_train_w = acc(y_train, pre_train_w)
        acc_dev_w = acc(y_dev, pre_dev_w)
        # for w_bar
        pre_train_w_bar = pre(w_bar, X_train)
        pre_dev_w_bar = pre(w_bar, X_dev)
        acc_train_w_bar = acc(y_train, pre_train_w_bar)
        acc_dev_w_bar = acc(y_dev, pre_dev_w_bar)
        acc_records.append(np.array([t, acc_train_w, acc_dev_w, acc_train_w_bar,acc_dev_w_bar ]))
    return w, w_bar, np.array(acc_records)

##############
# (a)
##############
print("For part 1 (apart2a.py)")
w, w_bar, acc_records  = avg_per_acc(X_train_np, y_train_np, X_dev_np, y_dev_np, maxiter = 100)
if 0:
    plt.figure(figsize=(8,8))
    plt.plot(acc_records[:,0], acc_records[:,1],c = 'r', label = "train accuracy with online perceptron")
    plt.plot(acc_records[:,0], acc_records[:,2],c = "b", label = "validation accuracy with online perceptron")
    plt.plot(acc_records[:,0], acc_records[:,3], c = 'g', label = "train accuracy with average perceptron")
    plt.plot(acc_records[:,0], acc_records[:,4],label = "validation accuracy with average perceptron")
    plt.xlabel("iter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



##############
# (c)
##############
print("For part 1 (c)")


def avg_per_acc_dev(X_train, y_train, X_dev, y_dev, maxiter = 100):
    acc_records = []
    num_instances, num_feats = X_train.shape
    w = np.zeros(num_feats)
    w_bar = np.zeros(num_feats)
    s = 1
    t = 0
    while t < maxiter:
        for i in range(num_instances):
            xi = X_train[i]
            yi = y_train[i]
            h = np.dot(xi,w) *yi
            if h <= 0:
                w = w + xi * yi

            w_bar = (s*w_bar + w)/(s+1)
            s = s + 1
        t = t+1
        if t%100 ==0:
            print("iter {}".format(t))
            print(t, acc_dev_w_bar)
        y_train = np.squeeze(y_train)
        y_dev = np.squeeze(y_dev)
        pre_dev_w_bar = pre(w_bar, X_dev)
        acc_dev_w_bar = acc(y_dev, pre_dev_w_bar)
        acc_records.append(np.array([t, acc_dev_w_bar ]))
    return w, w_bar, np.array(acc_records)



w, w_bar, acc_records_dev  = avg_per_acc_dev(X_train_np, y_train_np, X_dev_np, y_dev_np, maxiter = 1000)

plt.figure(figsize=(8,8))
plt.plot(acc_records_dev[:,0], acc_records_dev[:,1],label = "validation accuracy with average perceptron")
plt.xlabel("iter")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
