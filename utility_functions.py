import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import plotly.express as px
import plotly.graph_objects as go

from pmdarima.arima import auto_arima

from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

def run_diagnostics(ts, title, label):
    '''
    Function to plot the time series, decomposition, autocorrelation and partial autocorrelation functions of the
    original time series, the time series when differenced, and the time series when twice-differenced.
    
    Parameters:
    ts - Time series to be analyzed.
    title - Title for plots.
    label - Label for y-axis on plots.
    
    Returns:
    Plot of time series
    Plot of decomposition of time series
    Plots of autocorrelation and partial autocorrelation functions for time series
    P value output of adfuller test on time series
    
    Plot of differenced time series
    Plot of decomposition of differenced time series
    Plots of autocorrelation and partial autocorrelation functions for differenced time series
    P value output of adfuller test on differenced time series
    
    Plot of twice-differenced time series
    Plot of decomposition of twice-differenced time series
    Plots of autocorrelation and partial autocorrelation functions for twice-differenced time series
    P value output of adfuller test on twice-differenced time series
    '''
    
    # Define differenced time series
    diff_ts = ts.diff().dropna()
    
    # Define twice-differenced time series
    diff_diff_ts = diff_ts.diff().dropna()
    
    # Plot Initial time series, applies title and label
    ts.plot()
    plt.title('{}'.format(title))
    plt.xlabel('Date')
    plt.ylabel('{}'.format(label))
    plt.show()
    
    # Plots decomposition of time series
    decomposed_ts = sd(ts)
    decomposed_ts.plot()
    plt.show()
    
    # Plots acf and pacf for time series
    plot_acf(ts)
    plt.show()
    plot_pacf(ts)
    plt.show()

    # Prints the p-value of an adfuller test on original time series
    adfuller_ts = adfuller(ts)
    print('Adfuller results - p value:', adfuller_ts[1])
    
    # Plots differenced time series
    diff_ts.plot()
    plt.title('Differenced {}'.format(title))
    plt.xlabel('Date')
    plt.ylabel('{}'.format(label))
    plt.show()
    
    # Plots decomposition of differenced time series
    decomposed_diff = sd(diff_ts)
    decomposed_diff.plot()
    plt.show()
    
    # Plots acf and pacf of differenced time series
    plot_acf(diff_ts)
    plt.show()
    plot_pacf(diff_ts)
    plt.show()
    
    # Prints p value of adfuller test on differenced time series
    adfuller_diff = adfuller(diff_ts)
    print('Differenced adfuller results - p value:', adfuller_diff[1])
    
    # Plots twice-differenced time series
    diff_diff_ts.plot()
    plt.title('Twice Differenced {}'.format(title))
    plt.xlabel('Date')
    plt.ylabel('{}'.format(label))
    plt.show()
    
    # Plots decomposition of twice-differenced time series
    decomposed_diff_diff = sd(diff_diff_ts)
    decomposed_diff_diff.plot()
    plt.show()
    
    # Plots acf and pacf of twice-differenced time series
    plot_acf(diff_diff_ts)
    plt.show()
    plot_pacf(diff_diff_ts)
    plt.show() 
    
    # Prints p value of adfuller test on twice-differenced time series
    adfuller_diff_diff = adfuller(diff_diff_ts)
    print('Twice-differenced adfuller results - p value:', adfuller_diff_diff[1])
    
def plot_test_predictions(yhat, train, test):
    '''
    Calculates root mean square error between predictions and test set, then plots the predictions against 
    observed values
    
    Parameters:
    yhat - model predictions
    train - training dataset
    test - test dataset
    
    Returns:
    Printout of rmse and a plot of predicted values'''
    
    rmse = sqrt(mean_squared_error(yhat, test))
    print('Root Mean Squared Error: ', round(rmse, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index,
                             y=train,
                             mode = 'lines', 
                             name='Total Cases'))
    fig.add_trace(go.Scatter(x=test.index,
                             y=test,
                             mode = 'lines', 
                             name='Total Cases'))
    fig.add_trace(go.Scatter(x=test.index,
                             y=yhat,
                             mode='lines',
                             name='Predicted Total Cases'))


    fig.show()

def plot_forecast(predictions, ts):
    '''
    Plots observed and forecasted values
    
    Parameters:
    predictions - model forecast
    ts - original time series
    
    Returns:
    Plot of predicted values
    '''
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=ts.index,
                             y=ts,
                             mode = 'lines', 
                             name='Total Cases'))
    
    fig.add_trace(go.Scatter(x = predictions.index,
                             y=predictions,
                             mode='lines',
                             name='Predicted Total Cases'))

    fig.show()