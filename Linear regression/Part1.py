import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd

lr = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
x_norm, y_norm, N = gd.get_norm_data('PA1_train.csv')
X_norm_dev, y_norm_dev, N_dev = gd.get_norm_data('PA1_dev.csv')
epochs = 10000
lamda=0
final_w, ttl_MSE, iter_list, N= gd.get_list_withoutlamda(x_norm, y_norm, lr, epochs, N)

count_1 = 0
for i in ttl_MSE:
    print('iterations:{}'.format(iter_list))
    print('learning_rate:{}(traning data)'.format(lr[count_1]))
    print('mse:{}'.format(i[-1]))
    count_1+=1

count_2 = 0
for j in final_w:
    print('learning_rate:{}(validation data)'.format(lr[count_2]))
    mse = (X_norm_dev.dot(j)- y_norm_dev) **2/N_dev
    print('mse:{}'.format(mse.sum()))
    count_2+=1
