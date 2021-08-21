import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd

lr = [100, 10, 1e-1, 1e-2, 1e-3, 1e-4, 1e-10,1e-15]
df = gd.get_data('PA1_train.csv')
y_nonorm = df['price'].values.reshape(-1, 1)
x_nonorm = df.drop(columns = ['price']).values
epochs = 10000
lamda = 0.001
N=int(5598)
final_w, ttl_MSE, iter_list, N= gd.get_list_withlamda(x_nonorm, y_nonorm, lr, epochs, lamda, N)

count_1 = 0
count_2 = 0
for j in ttl_MSE:
    print('learning_rate:{}(traning data)'.format(lr[count_1]))
    print('final mse:{}'.format(j[-1][-1]))
    count_1+=1

df_ = gd.get_data('PA1_dev.csv')
y_val = df['price'].values.reshape(-1, 1)
X_val = df.drop(columns = ['price']).values

for i in final_w:
    print('learning_rate:{}(validation data'.format(lr[count_2]))
    mse = (X_val.dot(i)- y_val) **2/N
    print('final mse:{}'.format(mse.sum()))
    count_2 += 1
