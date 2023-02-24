# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:24:51 2023

@author: Mostafa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


sns.set(rc={'figure.figsize': [7, 7]}, font_scale=1.2)

df = pd.read_csv('Salary_Data.csv')

print(df)
print(df.describe())
sns.jointplot(data= df, x= 'YearsExperience', y ='Salary' )

x=df['YearsExperience'].values.reshape(-1, 1)
y=df['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train)

model = LinearRegression()
print(model.fit(x_train, y_train))

y_predict= model.predict(x_test)
print(y_predict)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print(model.coef_)
print(model.intercept_)



plt.plot(x_train, y_train, 'ro', label='training_data')
plt.plot(x_test, y_test, 'bo', label='testing_data')
plt.plot(x_test, y_predict, 'g-', label='predicted_data')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()