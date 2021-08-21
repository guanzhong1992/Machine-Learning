import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd


pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

def three_features(data):
    three_features_data = data[['waterfront', 'grade', 'condition']]
    waterfront_data = three_features_data['waterfront'].value_counts().sort_index(ascending=True)
    grade_data = three_features_data['grade'].value_counts().sort_index(ascending=True)
    condition_data = three_features_data['condition'].value_counts().sort_index(ascending=True)
    waterfront_data_perc = pd.Series(waterfront_data.values / waterfront_data.values.sum())
    grade_data_perc = pd.Series(grade_data.values / grade_data.values.sum(), index = list(range(4,14)))
    condition_data_perc = pd.Series(condition_data.values / condition_data.values.sum(), index = list(range(1,6)))
    three_features_table = pd.concat([waterfront_data_perc,grade_data_perc,condition_data_perc],axis = 1)
    three_features_table.columns = ['waterfront', 'grade', 'condition']

    return three_features_table

def other_features(data):
    other_list = [i for i in data if i not in ['waterfront', 'grade', 'condition']]
    other_feartures = data[other_list]
    other_feartures_mean = other_feartures.mean()
    other_feartures_std = other_feartures.std()
    other_feartures_range = other_feartures.max() - other_feartures.min()
    other_feartures_table = pd.concat([other_feartures_mean, other_feartures_std, other_feartures_range], axis = 1)
    other_feartures_table.columns = ['mean', 'std', 'range']

    return other_feartures_table


df = gd.get_data('PA1_train.csv')
X_df = df.drop(columns=['price','dummy'])
other_feartures = other_features(X_df)
three_features_data = three_features(X_df)
print(other_feartures.T)
print(three_features_data.T)
