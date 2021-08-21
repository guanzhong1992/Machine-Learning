import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_norm_data(name):
    df =pd.read_csv(name)
    del df['id']
    df['day'] = pd.DatetimeIndex(df['date']).day
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    del df['date']
    N = int(5598)
    y_norm = df['price'].values.reshape(-1, 1)
    x_norm = df.drop(columns=['price', 'dummy'])
    x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
    X_1 = np.ones(len(x_norm)).reshape(-1, 1)
    x_norm = np.concatenate((X_1, x_norm), axis=1)

    return x_norm, y_norm, N

def get_data(name):
    df = pd.read_csv(name)
    del df['id']
    df['day'] = pd.DatetimeIndex(df['date']).day
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    del df['date']

    return df

def gradient_descent(X, y, learning_rate, epochs, lamda, N):

    def get_mse(X, w, y,lamda,N):
        y_i = X.dot(w)
        mse = ((y_i - y) ** 2)/N
        re = lamda * (w ** 2)
        mse = mse.sum() + re.sum()/N
        return mse

    def get_gradients(X, w, y, lamda):
        y_i = X.dot(w)
        y_reshaped = y.reshape(-1, 1)
        grads = 2 * (X.T.dot(y_i - y_reshaped)) + 2 * lamda * w
        return grads

    n = X.shape[1]
    w = np.random.rand(n).reshape(-1, 1)
    w_history = w
    mse_history = np.array([get_mse(X, w, y, lamda, N)])
    n_prints = epochs // 10
    iterations=0
    iter_list=[]

    for i in range(epochs):
        grads = get_gradients(X, w, y, lamda)
        w -= learning_rate * grads
        mse = get_mse(X, w, y, lamda, N)
        w_history = np.hstack([w_history, w])
        mse_history = np.append(mse_history, mse)

        if i % n_prints == 0:
            print("w: {}; mse: {}".format(w.ravel(), mse))
        if mse > 10e300:
            iterations=i
            iter_list.append(iterations)
            break;
        if abs(mse) <= 0.5:
            iterations=i
            iter_list.append(iterations)
            break

    return w, mse_history, w_history, iter_list

def LinearRegression(x_norm, y_norm, lr, epochs, N, lamda=0):
    W_list = []
    MSE_list = []
    W_history_list = []
    if lamda == 0:
        for i in lr:
            print('======Learning Rate:{} ======'.format(i))
            w, mse_history, w_history,iter_list, = gradient_descent(x_norm, y_norm, i, epochs, lamda, N)
            W_list.append(w)
            MSE_list.append(mse_history)
            W_history_list.append(w_history)
    else:
        for i in lr:
            print('======Learning Rate:{} with lamda:{}======'.format(i,N))
            w, mse_history, w_history,iter_list = gradient_descent(x_norm, y_norm, i, epochs, lamda, N)
            W_list.append(w)
            MSE_list.append(mse_history)
            W_history_list.append(w_history)

    return W_list, MSE_list, W_history_list, iter_list, N

def lr_plt(data):
    for i in data:
        plt.plot(range(i.size), i)

def get_list_withoutlamda(x_norm, y_norm, lr, epochs, N):
    W_list, MSE_list, W_history_list, iter_list,N = LinearRegression(x_norm, y_norm, lr, epochs, N)
    for i in range(8):
        lr_plt(MSE_list[i:i+1])
        plt.legend(['1e-{}'.format(i)])
        plt.xlabel('iterations')
        plt.ylabel('mse')
        plt.show()

    return W_list, MSE_list, iter_list ,N

def get_list_withlamda(x_norm, y_norm, lr, epochs, lamda, N):
    W_r2, MSE_list_r2, W_history_list_r2, iter_list, N = LinearRegression(x_norm, y_norm, lr, epochs, lamda, N)
    MSE_1 = MSE_list_r2[:8]
    MSE_2 = MSE_list_r2[8:16]
    MSE_3 = MSE_list_r2[16:24]
    MSE_4 = MSE_list_r2[24:32]
    MSE_5 = MSE_list_r2[32:40]
    MSE_6 = MSE_list_r2[40:]
    ttl_MSE = [MSE_1, MSE_2, MSE_3, MSE_4, MSE_5, MSE_6]
    for j in range(len(lr)):
        lr_plt(MSE_list_r2[j:j+1])
        plt.xlabel('iterations')
        plt.ylabel('mse')
        plt.title('lr={}'.format(lr[j]))
        plt.show()


    return W_r2, ttl_MSE, iter_list, N
