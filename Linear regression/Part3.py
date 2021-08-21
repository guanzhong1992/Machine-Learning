import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd

df = gd.get_data('PA1_train.csv')
df["room_radio"]=df["bedrooms"]/df["bathrooms"]
plt.bar(df["room_radio"], df["price"])
plt.show()
df["sqft_outside"]=df["sqft_lot"]/df["sqft_living"]
plt.bar(df["sqft_outside"], df["price"])
plt.show()
df["bathroom_per"]=df["bathrooms"]/df["floors"]
plt.bar(df["bathroom_per"], df["price"])
plt.show()
