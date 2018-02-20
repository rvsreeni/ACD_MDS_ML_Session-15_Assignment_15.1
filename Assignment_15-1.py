#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 19:15:55 2018

@author: macuser
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data, columns=boston.feature_names)

#print(boston.DESCR)
bos['PRICE'] = boston.target
print(bos.head())

# LR model - house price Vs per capita Crime rate 
# create X and y
feature_cols = ['CRIM']
X = bos[feature_cols]
y = bos.PRICE
print('\n LR model - House price Vs Crime rate') 
# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)
bos.plot(kind='scatter', x='CRIM', y='PRICE')
X_new = pd.DataFrame({'CRIM': [bos.CRIM.min(), bos.CRIM.max()]})
preds = lm.predict(X_new)
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()
# print intercept and coefficients
print('Intercept ',lm.intercept_)
print('Coefficient ',lm.coef_)

# LR model - house price Vs Age of house
# create X and y
feature_cols = ['AGE']
X = bos[feature_cols]
y = bos.PRICE
print('\n LR model - House price Vs Age of house') 
# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)
bos.plot(kind='scatter', x='AGE', y='PRICE')
X_new = pd.DataFrame({'AGE': [bos.AGE.min(), bos.AGE.max()]})
preds = lm.predict(X_new)
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()
# print intercept and coefficients
print('Intercept ',lm.intercept_)
print('Coefficient ',lm.coef_)

# LR model - house price Vs Distance to Boston employment 
# create X and y
feature_cols = ['DIS']
X = bos[feature_cols]
y = bos.PRICE
print('\n LR model - House price Vs Distance to Boston employment') 
# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)
bos.plot(kind='scatter', x='DIS', y='PRICE')
X_new = pd.DataFrame({'DIS': [bos.DIS.min(), bos.DIS.max()]})
preds = lm.predict(X_new)
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()
# print intercept and coefficients
print('Intercept ',lm.intercept_)
print('Coefficient ',lm.coef_)

# LR model - house price Vs crime rate, age, distance 
# create X and y
feature_cols = ['CRIM', 'AGE', 'DIS']
X = bos[feature_cols]
y = bos.PRICE
# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)
print('\n\n LR model - House price Vs (Crime + Age + Distance)')
# print intercept and coefficients
print('Intercept ',lm.intercept_)
print('Coefficient ',lm.coef_)