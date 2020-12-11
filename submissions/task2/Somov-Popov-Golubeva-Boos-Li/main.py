import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
rcParams['figure.figsize'] = 15, 5
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from sklearn.metrics import r2_score


import datetime
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from pandas.core.nanops import nanmean as pd_nanmea

output_notebook()

training_data = pd.read_excel('training.xlsx')
date = training_data["Date"]
value = training_data["Value"]
training_data.index  = training_data.Date
training_data = training_data.drop('Date', axis=1)
training_data.sort_values('Date', ascending = True)
training_data.Value.plot( figsize=(10,6), fontsize=14, color = 'black')

testing_data = pd.read_excel('testing.xlsx')
tdate = testing_data["Date"]
tvalue = testing_data["Value"]
testing_data.index  = testing_data.Date
testing_data = testing_data.drop('Date', axis=1)
testing_data.sort_values('Date', ascending = True)
testing_data.Value.plot( figsize=(10,6), fontsize=14, color = 'red')

# %%
X_training = pd.read_excel("training.xlsx").values.copy()
X_testing  = pd.read_excel("testing.xlsx").values.copy()

# %%
p = figure(title = 'Data', x_axis_label = 'date', y_axis_label = 'value', x_axis_type='datetime', width = 1000)
p.line(date, value, color = 'black')
p.line(date, value.rolling(30, center = True).mean(), color = 'red')
p.line(date, value.rolling(5, center = True).std(), color = 'purple')

# %%
show(p)

# %%
d = figure(title = 'Data', x_axis_label = 'date', x_axis_type='datetime', width = 1000)
difference = training_data.Value.shift(-1)-training_data.Value

d.line(date, difference, color = 'black',  width = 1)
d.line(date, difference.rolling(30, center=True).mean(), color = 'red', width = 3)
show(d)

# %%
decompos = sm.tsa.seasonal_decompose(training_data.Value, model="additive")

# %%
#Additive model
rcParams['figure.figsize'] = 10, 6
decompos = sm.tsa.seasonal_decompose(training_data.Value, model="additive")
decompos.plot()
plt.show()

# %%
print('Аддитивный тренд:')
decompos.trend.dropna(inplace=True)
sm.tsa.stattools.adfuller(decompos.trend)

# %%
print('Аддитивная сезонность:')
decompos.seasonal.dropna()
sm.tsa.stattools.adfuller(decompos.seasonal)

# %%
print('Остаток:')
decompos.resid.dropna(inplace=True)
sm.tsa.stattools.adfuller(decompos.resid)

# %%
decompos = sm.tsa.seasonal_decompose(training_data.Value, model="multiplicate")
decompos.plot()
plt.show()

# %%
print('Мультипликативный тренд:')
decompos.trend.dropna(inplace=True)
adfuller(decompos.trend)

# %%
print('Мультипликативная сезонность:')
decompos.seasonal.dropna()
adfuller(decompos.seasonal)

# %%
print('Мультипликативный остаток:')
decompos.resid.dropna(inplace=True)
adfuller(decompos.resid)

# %%
adfuller(training_data.Value)

#build autocorrelation and partial autocorrelation
traindiff = training_data.Value.diff(periods=1).dropna()
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(traindiff.values.squeeze(), lags=49, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(traindiff, lags=25, ax=ax2)


# %%
def arima(train, order, test):
    ar = figure(title = 'Data', x_axis_label = 'date', x_axis_type='datetime', width = 1000)
    
    model = sm.tsa.ARIMA(training_data.Value, order=order, freq='MS').fit()
    y_r = model.predict(start = training_data.shape[0], end = training_data.shape[0] + testing_data.shape[0]-1)
    
    ar.line(date, value, color = 'black')
    ar.line(tdate, tvalue, color = 'red')
    ar.line(testing_data.index, y_r, color = 'green')
    print("r2: ", r2_score(testing_data.Value, y_r))
    show(ar)


# %%
arima(training_data, (1,0,3), testing_data)
