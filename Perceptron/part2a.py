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

def pre(w, X):
    ans =  np.matmul(X, w)
    return np.sign(ans)

def acc(gt_y, pre_y):
    num = pre_y.shape
    pre_correct = np.sum((pre_y == gt_y)*1)
    return np.squeeze(pre_correct/num)



def get_w_from_a(a, y, X):
    ay = (a*y).reshape(-1, 1)
    _, dims = X.shape
    ayy = np.repeat(ay, dims, axis = 1)
    ans = np.sum(ayy * X, axis = 0)
    return ans

import time
def kernel_percep(X_train, y_train, X_dev, y_dev, maxiter = 100, p = 1):

    records = []
    num_instances, num_feats = X_train.shape
    a = np.zeros(num_instances)
    K0 = np.dot(X_train, X_train.T)
    K = np.power(K0, p)
    print("Finished the computation of kernel.")
    # iteration process
    t = 0
    while t < maxiter:
        start_time = time.time()
        for i in range(num_instances):
            xi = X_train[i]
            yi = y_train[i]
            ki = K[i]
            assert a.shape == ki.shape == y_train.shape
            u = np.sum(a * ki * y_train)
            if u*yi <= 0:
                a[i] = a[i] + 1
        t += 1
        w = get_w_from_a(a, y_train, X_train)
        pre_train = pre(w, X_train)
        pre_dev = pre(w, X_dev)
        acc_train = acc(y_train, pre_train)
        acc_dev = acc(y_dev, pre_dev)
        end_time = time.time()
        exp_time = end_time - start_time
        records.append(np.array([t, acc_train, acc_dev,exp_time ]))
        if t%10 ==0:
            print("iter {}".format(t))
            print(t, acc_train, acc_dev )

    return a, w , np.array(records)


##############
# (a)
##############
ans = kernel_percep(X_train_np,
                    y_train_np,
                    X_dev_np,
                    y_dev_np,
                    maxiter = 100,
                    p = 1)

if 0:
    plt.figure(figsize=(12,8))
    p = 1
    a, w ,acc_records_kernel = ans
    plt.plot(acc_records_kernel[:,0], acc_records_kernel[:,1],  c = 'r',         label = "train accuracy (p = {})".format(p))
    plt.plot(acc_records_kernel[:,0],  acc_records_kernel[:,2], c = 'b',         label = "validation accuracy (p = {})".format(p))
    plt.xlabel("iter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


##############
# (b)
##############
ans_dict = {}
for p in [1, 2, 3, 4, 5]:
    print("begin p : {}".format(p))
    ans = kernel_percep(X_train_np,
                        y_train_np,
                        X_dev_np,
                        y_dev_np,
                        maxiter = 100,
                        p = p)
    ans_dict[p] = ans

if 0:
    plt.figure(figsize=(12,8))
    for p in [1, 2, 3, 4, 5]:
        a, w ,acc_records_kernel = ans_dict[p]
        plt.plot(acc_records_kernel[:,0], acc_records_kernel[:,1],              label = "train accuracy (p = {})".format(p))
        plt.plot(acc_records_kernel[:,0], acc_records_kernel[:,2],              label = "validation accuracy (p = {})".format(p))
    plt.xlabel("iter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


##############
# part2a (c)
##############

best_valid_acc = []
p_lst = [1, 2, 3, 4, 5]
for p in p_lst:
    a, w ,acc_records_kernel = ans_dict[p]
    best_valid_acc.append(np.max(acc_records_kernel[:,2]))

if 0:
    plt.figure()
    plt.scatter(p_lst, best_valid_acc)
    plt.xlabel("p")
    plt.ylabel("Best validation accuracy")
    plt.title("Best validation accuracy versus p")
    plt.show()


##############
# part 2a (d)
##############

plt.figure(figsize=(8,8))
p = 1
a, w ,acc_records_kernel = ans_dict[p]
iterations = acc_records_kernel[:,0]
time_each_iter = acc_records_kernel[:,-1]
time_cumsum = np.cumsum(time_each_iter)
plt.plot(iterations, time_cumsum)
plt.xlabel("iterations")
plt.ylabel("Empirical runtime")
plt.title("Empirical runtime versus iterations ")
plt.show()
