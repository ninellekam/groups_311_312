#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import product
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.iolib.table import SimpleTable

from pylab import rcParams
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 16),
         'axes.labelsize': 22,
         'axes.titlesize':22,
         'xtick.labelsize':22,
         'ytick.labelsize':22}
rcParams.update(params)


def multiplicative(data):
    result = sm.tsa.seasonal_decompose(data.Value, model='multiplicative')
    result.plot()
    plt.show()
    
    from random import randrange
    print("\nCheck the components:")
    print('Trend:')
    result.trend.dropna(inplace=True)
    adfuller2(result.trend)

    print('')
    print('Seasonal:')
    result.seasonal.dropna()
    adfuller2(result.seasonal)

    print('')
    print('Resid:')
    result.resid.dropna(inplace=True)
    adfuller2(result.resid)
    
    series = [i**2.0 for i in range(1,100)]
    result = sm.tsa.seasonal_decompose(series, model='multiplicative', period=1)
    rcParams.update(params)
    result.plot()
    plt.show()

def additive(data):
    result = sm.tsa.seasonal_decompose(data.Value, model='additive')
    result.plot()
    plt.show()
    
    from random import randrange
    print("\nCheck the components:")
    print('Trend:')
    result.trend.dropna(inplace=True)
    adfuller2(result.trend)

    print('')
    print('Seasonal:')
    result.seasonal.dropna()
    adfuller2(result.seasonal)

    print('')
    print('Resid:')
    result.resid.dropna(inplace=True)
    adfuller2(result.resid)
    
    series = [i+randrange(10) for i in range(1,100)]
    result = sm.tsa.seasonal_decompose(data.Value, model='additive', period=1)
    result.plot()
    plt.show()

def adfuller2(df):

    print(" > Is the data stationary ?")
    dftest = adfuller(df)
    print("Test statistic = {:.3f}".format(dftest[0]))
    print("P-value = {:.3f}".format(dftest[1]))
    print("Critical values :")
    for k, v in dftest[4].items():
        print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))

training_data = pd.read_excel('~/prac/training.xlsx')
#print(training_data)
training_data.index  = training_data.Date
training_data = training_data.drop('Date', axis=1)
training_data.sort_values('Date', ascending = True)
training_data.Value.plot()
#print(training_data.Value)

adfuller2(training_data)

additive(training_data)

multiplicative(training_data)


dataset = pd.read_excel('~/prac/training.xlsx', index_col='Date')
datatest = pd.read_excel('~/prac/testing.xlsx', index_col='Date')

time_series = dataset.Value
time_series.plot(figsize=(12,6))

# Проверка исходного ряда на стационарность
test = sm.tsa.adfuller(time_series)
print('adf: ', test[0]) 
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0] > test[4]['5%']: 
    print('есть единичные корни, ряд не стационарен')
else:
    print('единичных корней нет, ряд стационарен')

# Нахождение порядка интегрированности ряда
time_series_diff = time_series.diff(periods=1).dropna()
# Проверка разности на стационарность
test = sm.tsa.adfuller(time_series_diff)
print('adf: ', test[0])
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0] > test[4]['5%']: 
    print('есть единичные корни, ряд не стационарен')
else:
    print('единичных корней нет, ряд стационарен')

time_series_diff.plot(figsize=(12,6))

# ACF(авторкорреляционная) & PACF(частично автокорреляционная) функции
# для определения параметров q и p соотвественно     
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(time_series_diff, lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(time_series_diff, lags=25, ax=ax2)

model = ARIMA(dataset.Value, order=(0, 1, 4)).fit()
print ('AIC =', model.aic)

plt.figure(figsize=(15,7))
datatest.Value.plot()
pred = model.predict(start='1988-12-01', end='1993-12-01', typ = 'levels', dynamic = True)
print ('r2_score = ',r2_score(datatest.Value, pred['1989-01-01':]))
pred.plot()
plt.show()

model = ARIMA(dataset.Value, order=(0, 1, 5)).fit()
print ('AIC =', model.aic)

plt.figure(figsize=(15,7))
datatest.Value.plot()
pred = model.predict(start='1988-12-01', end='1993-12-01', typ = 'levels', dynamic = True)
print ('r2_score = ',r2_score(datatest.Value, pred['1989-01-01':]))
pred.plot()
plt.show()

# Подбор лучших параметров по критерию Акаике
qs = range(0, 20)
ps = range(0, 2)
d = 1

parameters = product(ps, qs)
parameters_list = list(parameters)

results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')

for param in parameters_list:
    try:
        model = ARIMA(dataset.Value, order=(param[0], d, param[1])).fit()
    except:
        print ('warning parametrs:', param)
        continue
    print ('Computing...')
    if model.aic < best_aic:
        best_model = model
        best_aic = model.aic
        best_param = param
        print (param, model.aic)
    results.append([param, model.aic])
    

warnings.filterwarnings('default')

result_table = DataFrame(results)
result_table.columns = ['parameters', 'aic']
print (result_table.sort_values(by = 'aic', ascending = [True]).head())






