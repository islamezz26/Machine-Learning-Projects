# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:10:49 2023

@author: Mostafa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set(rc={'figure.figsize': [7, 7]}, font_scale=1.2)

df = pd.read_csv('Ecommerce Customers.csv')
print(df.info())
print(df.describe())
print(df.columns)

sns.pairplot(data=df)

x= df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']

x_train,x_test,y_train,y_test= train_test_split(x,y,random_state= 42, test_size = .3 )

model = LinearRegression()
print(model.fit(x_train,y_train))

YPredict = model.predict(x_test)

print(YPredict)

model.score(x_train, y_train)
model.score(x_test, y_test)

print(model.intercept_)
print(model.coef_)

