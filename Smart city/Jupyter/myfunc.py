# =============================*BLOC - Importing packpages*=============================
# Common packages
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import datetime
import warnings
import requests

# Statistic module packages
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson

# Scikit_learn and RNN module packages
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Install inexistant packages
import importlib
if importlib.util.find_spec("missingno") is None:
  !pip install --user --upgrade missingno
if importlib.util.find_spec("pmdarima") is None:
  !pip uninstall numpy
  !pip uninstall pmdarima
  !pip install numpy
  !pip install pmdarima
  
import missingno as msno
import pmdarima as pm
# =============================*END - Importing packpages*=============================


# =============================*BLOC - General functions*=============================
def inspect_data(df=None, visualize_missing_value=False, display_form='matrix', freq=None, figsize=(6,4)):
  """Function to display the information of data
  - Paramters:
    df: DataFrame or Series
    visualize_missing_value: is or not to visualize the missing value, default False
    display_form: visualization chart to use, 'matrix'(defaul), 'bar', 'heatmap'
    figsize: The size of the figure to display
    freq: Specify a periodicity for time-series data, default 'D'  
  - Return value: None
  """
  print(df.info())
  print(">>> The first five lines of data: \n", df.head())
  if visualize_missing_value == True:
    print(">>> Missing value visualization:")
    if display_form == 'matrix':
      if freq is not None:
        msno.matrix(df, freq=freq, figsize=figsize)
      else:
        msno.matrix(df, figsize=figsize)
    if display_form == 'bar':
      msno.bar(df, figsize=figsize)
    if display_form == 'heatmap':
      msno.heatmap(df, figsize=figsize)
  plt.show()
  return


def plot_data(df=None, figsize=(12,8), layout=(3,3)):
  '''Function to plot the time series data
  - Paramters:
    df: DataFrame or Series
    figsize: size of figure
    layout: layout of figure plot decided by the number of time-series 
  - Return value: None  
  '''
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):  
    df = df.to_frame()
  df.plot(subplots=True, figsize=figsize, layout=layout, colormap='Dark2', fontsize=12)
  plt.show() 
  return


def fill_missing_values(df=None, method='ffill', interpolate_mode='linear', fill_value=0, figsize=(8,4)):
  """Function to fill the missing value of time series
  - Paramters:
    df: DataFrame or Series
    method: could be ['ffill'(default), 'bfill', 'value', 'interpolate']
    interpolate_mode: compulsory when methode='interpolate', could be ['linear'(defaut), 'polynomial', 'akima', 'pad', 'nearest', 'zero', 'quadratic', 'cubic', 'spline']
    fill_value: compulsory when methode='value', 0 by default
    figsize: size of figure
  - Return value:
    df_fill: series with missing value filled
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):  
    df = df.to_frame()
  # Filled by foreward or backward value
  if method in ['ffill', 'bfill']:   
    df_fill = df.fillna(method=method)
  # Filled by given a value
  if method == 'value':         
    df_fill = df.fillna(fill_value)
  # Fiiled by method 'interpolation'
  if method == 'interpolate':
    df_fill = df.interpolate(interpolate_mode, order=2)
  return df_fill


def conversion_to_ppb(df=None):
  """Function to convert polluant concentration in unit µg/m3 to ppb
    - µg/m3 is micrograms of gaseous pollutant per cubic meter of ambient air
    - ppb (v) is parts per billion by volume
    - NO2 1 ppb = 1.88 µg/m3
    - NO 1 ppb = 1.25 µg/m3
    - O3 1 ppb = 2.00 µg/m3
    - SO2: 1 ppb = 2.62 µg/m3
    - CO 1 ppm = 1 mg/m3
  - Parameters:
    df: DataFrame or Series
  - Return value:
    df: DataFrame converted
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Get all the column names
  column_names = df.columns.values
  for i, column_name in enumerate(column_names):
    # Case NO2
    if 'NO2' in column_name:
      df[column_name] = df[column_name] / 1.88
    # Case NOx
    if 'NOx' in column_name:
      df[column_name] = df[column_name] / 1.25
    # Case O3
    if 'O3' in column_name:
      df[column_name] = df[column_name] / 2.00
    # Case SO2
    if 'SO2' in column_name:
      df[column_name] = df[column_name] / 2.62
  return df


def select_period_resample(df=None, columns=[], start_at=None, end_at=None, freq=None):
  """Function to extract a given period of Time Series
  - Paramters:
    df: DataFrame or Series
    columns: list of dataframe columns to choose, defalt all
    strat_time: start time point, format String or Timestampe
    end_time: end time point, format Timestamp
    freq: set resampling frequency of data
  - Return value: 
    Series with period selected and resampled
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Transform datetime string to timestamp
  start_at = pd.to_datetime(start_at)
  end_at = pd.to_datetime(end_at)
  if len(columns) != 0: # Extract part of columns
    return df.loc[start_at:end_at][columns].resample(freq).mean()
  else: # Extract all the columns 
    return df.loc[start_at:end_at].resample(freq).mean()


def decompose_series(df=None, model='additive', return_seasonal=False, return_trend=False, figsize=(8,4)):
  """Function to decompose the series into trend, seasonal and residual, to return seasonal and trend DataFrame if needed
  - Paramters:
    df: DataFrame or Series
    model : Type of seasonal component, {"additive"(default), "multiplicative"}
    return_seasonal; is or not to return seasonal dataframe of all the columns
    return_trend; is or not to return trend dataframe of all the columns
    figsize: size of figure
  - Return value: 
    trend_df or seasonal_df
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Initialize dict to stock trend and seasoanl component
  my_dict = {}
  my_dict_trend = {}
  my_dict_seasonal = {}
  # Get all the column names
  column_names = df.columns
  for column_name in column_names:
    decomp = seasonal_decompose(df[column_name], model=model)
    my_dict[column_name] = decomp
    if return_seasonal==False and return_trend==False:
      decomp.plot()
      plt.title(column_name, fontsize=10, fontweight=4)
      plt.show()
  # Extract the trend component
  for column_name in column_names:
    my_dict_trend[column_name] = my_dict[column_name].trend
  # Extract the sesonal component
  for column_name in column_names:
    my_dict_seasonal[column_name] = my_dict[column_name].seasonal
  # Convert to a DataFtrame
  trend_df = pd.DataFrame.from_dict(my_dict_trend)
  seasonal_df = pd.DataFrame.from_dict(my_dict_seasonal)
  if return_seasonal==True and return_trend==False:
    return seasonal_df
  if return_seasonal==False and return_trend==True:
    return trend_df
  if return_seasonal==True and return_trend==True:
    return seasonal_df, trend_df
  if return_seasonal==False and return_trend==False:
    return


def plot_rolling_means_variance(df=None, window=None, figsize=(12,2)):
  '''Function to inpsect the stationarity of time series by visualizing rolling means and variances
  - Parameters:
    df: DataFrame or Series
    window: rolling size of window, equal to seasonal lags
    figsize: size of figures
  - Return value: None
   '''
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Get all the column names
  column_names = df.columns
  for i, column_name in enumerate(column_names):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[column_name], label='raw data')
    ax.plot(df[column_name].rolling(window=window).mean(), label="rolling mean")
    ax.plot(df[column_name].rolling(window=window).std(), label="rolling std")
    ax.set_xlabel('Time')
    ax.set_ylabel(column_name)
    ax.legend()
  plt.show()
  return


def is_stationnary(df=None):
  """Function to check if the time series is stationary by Augmented Dickey-Fuller Test
  - Paramters:
    df: DataFrame or Series
  - Return value: None
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Get all the column names
  column_names = df.columns.values
  for i, column_name in enumerate(column_names):
    print("Augmented Dickey-Fuller Test on {}".format(column_name))
    # Calculate ADF value
    results_ADF = adfuller(df.iloc[:,i].dropna(), autolag='AIC')
    print('Null Hypothesis: Data has unit root. Non-Stationary.')
    print("Test statistic = {:.3f}".format(results_ADF[0]))
    print("P-value = {:.3f}".format(results_ADF[1]))
    print("Critical values :")
    for k, v in results_ADF[4].items():
        print("\t{}: {:.3f} ==> The data is {} stationary with {}% confidence".format(k, v, "not" if v<results_ADF[0] else "", 100-int(k[:-1])))
    print('\n')
  return


def transform_diff_log(df=None, lags=1, rolling=False, windows=None, log=False, valid=False):
  """Function to transform a non-stationary series to a stationary one by differencing, or to remove the exponential trend
  - Paramters:
    df: DataFrame or Series
    lags: differences lags to take, 1 default
    rolling: rolling data to make it more smoothly
    windows: size of rolling window, compulsary if rolling=True
    log: logging the data if the raw data exhibits an exponential trend
    valid: is or not to verify the stationarity
  - Return value:
    df: stationary time series
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  if log:
    df = np.log(df).dropna()
  if rolling:
    df = df.rolling(window=windows).mean().dropna()
  if lags is not None:
    df = df.diff(lags).dropna()
  if valid:
    is_stationnary(df)
  return df


def acf_pacf_plot(df=None, lags=20, alpha=0.1, acf=True):
  """Funtion to visualize the ACF and PACF plot
  - Paramters:
    df: DataFrame or Series
    lags: numbers of lags of autocorrelation to be plotted
    alpha: set the width of confidence interval. if aplha=1, hidden blue bound
    acf: is or not to plot acf
  - Return value: None
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  if acf:
    fig, axes = plt.subplots(2,1)
    # Plot the ACF
    plot_acf(df, lags=lags, alpha=alpha, ax=axes[0])
    # Plot the PACF
    plot_pacf(df, lags=lags, alpha=alpha, ax=axes[1])
  else:
    fig, axes = plt.subplots()
    plot_pacf(df, lags=lags, alpha=alpha, ax=axes)
  plt.title(df.columns.values[0])
  plt.tight_layout()
  plt.show()
  return


def train_test_split(df=None, train_size=None, start_at=None, end_at=None):
  """Function to split a time series into training and testing set
  - Paramters:
    df: DataFrame or Series
    train_size: size of training set
    start_at: start point of training set, not to set if using train_size 
    end_at: end point of training set, not to set if using train_size 
  - Return value:
    training_set: training part of series
    testing_set: testing part of series
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  if train_size is not None:
    training_set = df.iloc[:np.int(len(df) * train_size)]
    testing_set = df.iloc[np.int(len(df) * train_size):]
  else:
    training_set = df.loc[start_at:end_at]
    testing_set = df.loc[end_at:]
  return training_set, testing_set


def data_labeled(df=None, freq=None, mode='Mean'):
  """Function to label the data based on the different thresholds of air pollution
  - Parameters:
    df: DataFrame or Series
    freq: data frequency by Hour('H') or by Day('D')
    mode: calculating hourly average(Mean, default) or maximum(Max) value
  - Remark:
    NO2, O3, SO2: hourly average/maximum value in µg/m³
    PM10, PM2.5: hourly average/maximum value or daily average adjusted in µg/m³
    CO: 8-hour rolling average/maximum in mg/m³
  - Return value:
    df: DataFrame labeled
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Resample and aggregate the data
  if mode == 'Max':
    df = df.resample(freq).max()
  else: 
    df = df.resample(freq).mean()
  # Get all the column names
  column_names = df.columns.values
  for i, column_name in enumerate(column_names):
    # Case NO2
    if 'NO2' in column_name:
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (53 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 53))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (100 - 54) * (df[column_name].iloc[np.where((df[column_name] >= 54) & (df[column_name] <= 100))].to_frame()  - 54) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (360 - 101) * (df[column_name].iloc[np.where((df[column_name] >= 101) & (df[column_name] <= 360))].to_frame()  - 101) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (649 - 361) * (df[column_name].iloc[np.where((df[column_name] >= 361) & (df[column_name] <= 649))].to_frame()  - 361) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (1249 - 650) * (df[column_name].iloc[np.where((df[column_name] >= 650) & (df[column_name] <= 1249))].to_frame()  - 650) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (2049 - 1250) * (df[column_name].iloc[np.where(df[column_name] >= 1250)].to_frame()  - 1250) + 301
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])
   # Case O3
    if 'O3' in column_name:
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (54 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 54))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (70 - 55) * (df[column_name].iloc[np.where((df[column_name] >= 55) & (df[column_name] <= 70))].to_frame() - 55) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (85 - 71) * (df[column_name].iloc[np.where((df[column_name] >= 71) & (df[column_name] <= 85))].to_frame()  - 71) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (106 - 86) * (df[column_name].iloc[np.where((df[column_name] >= 86) & (df[column_name] <= 105))].to_frame()  - 86) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (200 - 106) * (df[column_name].iloc[np.where((df[column_name] >= 106) & (df[column_name] <= 200))].to_frame()  - 106) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (300 - 201) * (df[column_name].iloc[np.where(df[column_name] >= 201)].to_frame()  - 201) + 201
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])
   # Case SO2
    if 'SO2' in column_name:
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (35 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 35))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (75 - 36) * (df[column_name].iloc[np.where((df[column_name] >= 36) & (df[column_name] <= 75))].to_frame()  - 36) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (185 - 76) * (df[column_name].iloc[np.where((df[column_name] >= 76) & (df[column_name] <= 185))].to_frame()  - 76) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (304 - 186) * (df[column_name].iloc[np.where((df[column_name] >= 186) & (df[column_name] <= 304))].to_frame()  - 186) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (604 - 305) * (df[column_name].iloc[np.where((df[column_name] >= 305) & (df[column_name] <= 604))].to_frame()  - 305) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (1004 - 605) * (df[column_name].iloc[np.where(df[column_name] >= 605)].to_frame()  - 605) + 301
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])
    # Case PM10
    if 'PM10' in column_name:
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (54 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 54))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (154 - 55) * (df[column_name].iloc[np.where((df[column_name] >= 55) & (df[column_name] <= 154))].to_frame()  - 55) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (254 - 155) * (df[column_name].iloc[np.where((df[column_name] >= 155) & (df[column_name] <= 1254))].to_frame()  - 155) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (354 - 255) * (df[column_name].iloc[np.where((df[column_name] >= 255) & (df[column_name] <= 354))].to_frame()  - 255) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (424 - 355) * (df[column_name].iloc[np.where((df[column_name] >= 355) & (df[column_name] <= 424))].to_frame()  - 424) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (604 - 425) * (df[column_name].iloc[np.where(df[column_name] >= 425)].to_frame()  - 425) + 301
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])  
    # Case PM2.5
    if 'PM2.5' in column_name:
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (12 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 12))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (35.4 - 12.1) * (df[column_name].iloc[np.where((df[column_name] >= 12.1) & (df[column_name] <= 35.4))].to_frame()  - 12.1) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (55.4 - 35.5) * (df[column_name].iloc[np.where((df[column_name] >= 35.5) & (df[column_name] <= 55.4))].to_frame()  - 35.5) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (150.4 - 55.5) * (df[column_name].iloc[np.where((df[column_name] >= 55.5) & (df[column_name] <= 150.4))].to_frame()  - 55.5) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (250.4 - 150.5) * (df[column_name].iloc[np.where((df[column_name] >= 150.5) & (df[column_name] <= 250.4))].to_frame()  - 150.5) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (500.4 - 250.5) * (df[column_name].iloc[np.where(df[column_name] >= 250.5)].to_frame()  - 250.5) + 301
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])
    # Case NOx
    if 'NOx' in column_name:
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (53 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 53))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (100 - 54) * (df[column_name].iloc[np.where((df[column_name] >= 54) & (df[column_name] <= 100))].to_frame()  - 54) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (360 - 101) * (df[column_name].iloc[np.where((df[column_name] >= 101) & (df[column_name] <= 360))].to_frame()  - 101) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (649 - 361) * (df[column_name].iloc[np.where((df[column_name] >= 361) & (df[column_name] <= 649))].to_frame()  - 361) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (1249 - 650) * (df[column_name].iloc[np.where((df[column_name] >= 650) & (df[column_name] <= 1249))].to_frame()  - 650) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (2049 - 1250) * (df[column_name].iloc[np.where(df[column_name] >= 1250)].to_frame()  - 1250) + 301
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])
    # Case CO
    if 'CO' in column_name:
      # 8-hour rolling average/maximum in mg/m³
      df_CO = df.copy()
      df_CO = df_CO.rolling(window=8).mean().fillna(method='bfill')
      df[column_name] = df_CO[column_name]
      df[str(column_name) + '_Class'] = None
      AQI_class1 = (50 - 0) / (4.4 - 0) * (df[column_name].iloc[np.where((df[column_name] >= 0) & (df[column_name] <= 4.4))].to_frame()  - 0) + 0
      index_class1 = AQI_class1.index
      AQI_class2 = (100 - 51) / (9.4 - 4.5) * (df[column_name].iloc[np.where((df[column_name] >= 4.5) & (df[column_name] <= 9.4))].to_frame()  - 4.5) + 51
      index_class2 = AQI_class2.index
      AQI_class3 = (150 - 101) / (12.4 - 9.5) * (df[column_name].iloc[np.where((df[column_name] >= 9.5) & (df[column_name] <= 12.4))].to_frame()  - 9.5) + 101
      index_class3 = AQI_class3.index
      AQI_class4 = (200 - 151) / (15.4 - 12.5) * (df[column_name].iloc[np.where((df[column_name] >= 12.5) & (df[column_name] <= 15.4))].to_frame()  - 12.5) + 151
      index_class4 = AQI_class4.index
      AQI_class5 = (300 - 201) / (30.4 - 15.5) * (df[column_name].iloc[np.where((df[column_name] >= 15.5) & (df[column_name] <= 30.4))].to_frame()  - 15.5) + 201
      index_class5 = AQI_class5.index
      AQI_class6 = (500 - 301) / (50.4 - 30.5) * (df[column_name].iloc[np.where(df[column_name] >= 30.5)].to_frame()  - 30.5) + 301
      index_class6 = AQI_class6.index
      df[str(column_name) + '_Class'].loc[index_class1] = 'Good'
      df[str(column_name) + '_Class'].loc[index_class2] = 'Moderate'
      df[str(column_name) + '_Class'].loc[index_class3] = 'Unhealthy for SG'
      df[str(column_name) + '_Class'].loc[index_class4] = 'Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class5] = 'Very Unhealthy'
      df[str(column_name) + '_Class'].loc[index_class6] = 'Hazardous'
      df[str(column_name) + '_AQI'] =  pd.concat([AQI_class1[column_name], AQI_class2[column_name], AQI_class3[column_name], AQI_class4[column_name], AQI_class5[column_name], AQI_class6[column_name]])
  # Fill the missing value
  df = df.fillna(method='ffill')
  return df


def Air_Situation_Dashbord(df=None, start_time=None, end_time=None, figsize=(10, 4), width=0.5, freq='H', mode='Mean', min_max_distance=0.1, return_AQI=False, title=None):
  """Function to display the dashbord of pollution concentration(sigle or more) for a seleted period
  - Parameters:
    df: DataFrame or Series
    start_time: start time point to display(String or Timestampe), not set when using whole data
    end_time: end time point to display(String or Timestampe), not set when using whole data
    figsize: size of figure, default (10,4)
    width: control the width between each bar, default 0.5
    freq: frequency of resampling, used in function 'data_labeled'
    mode: calculating hourly average(Mean, default) or maximum(Max) value
    min_max_distance: used to modeify the position of MIN and MAX value
    return_AQI: is or not to return labeled DataFrame
    title: title to set
  - Retrun value
    df: labeled DataFrame / None
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Transform datetime string to timestamp
  start_time = pd.to_datetime(start_time)
  end_time = pd.to_datetime(end_time)
  # Label data
  if start_time is not None and end_time is not None:
    df = data_labeled(df[start_time:end_time], freq=freq, mode=mode)
  else:
    df = data_labeled(df[start_time:end_time], freq=freq, mode=mode)
  # Get all the column names
  column_names = df.columns
  for i, column_name in enumerate(column_names):
    if column_name in ['NO2', 'NOx', 'O3', 'SO2', 'PM10', 'PM2.5', 'CO']:
      df_color = df.copy().dropna()
        # Caclulate the MIN and MAX value of polluation concentration in seleted period
      min_value = df_color[str(column_name) + '_AQI'].min()
      max_value = df_color[str(column_name) + '_AQI'].max()
        # Transform the polluation indice to color correpondent 
      df_color['Color'] = None
      index_green = df_color.iloc[np.where(df_color[str(column_name)+'_Class'] == 'Good')].index
      df_color['Color'].loc[index_green] = 'Green'
      index_yellow = df_color.iloc[np.where(df_color[str(column_name)+'_Class'] == 'Moderate')].index
      df_color['Color'].loc[index_yellow] = 'Yellow'
      index_orange = df_color.iloc[np.where(df_color[str(column_name)+'_Class'] == 'Unhealthy for SG')].index
      df_color['Color'].loc[index_orange] = 'Orange'
      index_red = df_color.iloc[np.where(df_color[str(column_name)+'_Class'] == 'Unhealthy')].index
      df_color['Color'].loc[index_red] = 'Red'
      index_purple = df_color.iloc[np.where(df_color[str(column_name)+'_Class'] == 'Very Unhealthy')].index
      df_color['Color'].loc[index_purple] = 'Purple'
      index_maroon = df_color.iloc[np.where(df_color[str(column_name)+'_Class'] == 'Hazardous')].index
      df_color['Color'].loc[index_maroon] = 'Maroon'
        # Set and plot the figure
      fig, ax = plt.subplots(figsize = figsize)
      ax.bar(x=df_color.index, height=df_color[str(column_name)+'_AQI'], width=width, color=df_color['Color'])
      ax.set_xlabel('Time')
      ax.set_ylabel(str(column_name) + "-AQI")
      ax.text(df.index[-1]+datetime.timedelta(days=min_max_distance), max_value, '-- MAX={:.1f}'.format(max_value), color='red')
      ax.text(df.index[-1]+datetime.timedelta(days=min_max_distance), min_value, '-- MIN={:.1f}'.format(min_value), color='blue')
      # Set the separating vertical lines
      for i in range(len(df.index)):
        if np.mod(i,6) == 0:
          ax.axvline(df.index[i], linewidth=1, linestyle='--', color='black', alpha=0.5)
          if i >= 24:
            i_temp = np.mod(i, 24)
            ax.text(df.index[i], max_value, i_temp)
          else:
            ax.text(df.index[i], max_value, i)
  plt.title(title, fontsize=18, loc='center')
  plt.show()
  if return_AQI:
    return df
  else:
    return


def evaluate(y_true=None, y_pred=None):
  """Evaluate MSE and RMS of true value and predict value
  - Parameters:
    y_true: Truth (correct) target values, DataFrame, Series or Array
    y_pred: Estimated target values same type with y_true
  - Return value: None
  """
  mse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)
  rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
  print("MSE Value:{}".format(mse))
  print("RMSE Value:{}".format(rmse))
  return


def is_incointegrated(P=None,Q=None):
  """Function to test if the linear combination(P - Coeff * Q) of two random series P and Q is not a random walk.
  - Paramters:
    P, Q: DataFrame or Series
  - Return value: 
    The incointegrated series if they are incointegrated
  """
  # Transform DataFrame object to Series objet
  if not isinstance(P, pd.Series):
    # P = P.iloc[:,0]
    P = pd.Series(P)
  if not isinstance(Q, pd.Series):
    Q = Q.iloc[:,0]
  # Regress P on Q to get the regression coefficient Coeff
  Q = sm.add_constant(Q)
  Coeff = sm.OLS(P,Q).fit().params[1]
  # Compute ADF
  inter_series = P - Coeff * Q.iloc[:,1]
  p_value = adfuller(inter_series)[1]
  if p_value < 0.05:
    print("Two series are incointegrated with with 95% confidence")
    return inter_series
  else:
    print("Two series are not incointegrated")
    return


def correlation_analyse(df1=None, df2=None, method='pearson', figsize=(5, 4)):
  """Function to calculate and display the correlation of two Time Series
  - Paramters:
    df1: DataFrame or Series
    df2: DataFrame or Series
    method: test the linear correlaiton by 'pearson'(default) or non-linear correlaiton by 'spearman'
    figsize: size of plot figure
  - Return value: None
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df1, pd.DataFrame):
    df1 = df1.to_frame()
  if not isinstance(df2, pd.DataFrame):
    df2 = df2.to_frame()
  # Concatenate dataframe
  if df2 is None:
    df = df1
  else:
    df = pd.concat([df1,df2], axis=1)
  # Calculate correlation coefficient
  correlation = df.corr(method=method)
  fig = sns.clustermap(correlation, annot=True, linewidth=0.1, figsize=figsize)
  plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=90)
  if method=='pearson':
    plt.title('Linear correlation analysis', fontsize=12, fontweight=5, loc='center')
  else:
    plt.title('Non-linear correlation analysis', fontsize=12, fontweight=5, loc='center')
  plt.tight_layout()
  plt.show()
  return

def extract_by_hour(df=None, start_day=None, end_day=None, start_time=None, end_time=None):
  '''Function to extract a specific time period of data for each day  
  - Parameters:
    df: DataFrame or Series
    start_day: Timestamp or string
    end_day: Timestamp or string
    start_time: start time point, int
    end_time: end time point, int
  - Return values:
    df: DataFrame with data in extracting period
  '''
  # Transform datetime string to Timestampe
  start = pd.to_datetime(start_day)
  end = pd.to_datetime(end_day)
  time_stamp = []
  for i in pd.date_range(start=start_day, end=end_day):
    for j in range(start_time, end_time+1):
      time_stamp.append(i + datetime.timedelta(hours=j))
  df = df.loc[time_stamp]
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  return df


def pc_last_previous(df=None):
  '''Function to calculates the % change between the last value and the mean of previous values
  - Paramters:
    df: DataFrame or Series
  - Return value:
    percent_change
  '''
  # Seperate the last value and all previous values into variables
  previous_values = df[:-1]
  last_values = df[-1]
  # Calculate the % of difference between the last value and the mean of earlier values
  percent_change = (last_values - np.mean(previous_values)) / np.mean(previous_values)
  return percent_change


def pc_rolling_plot(df=None, window=5, show=True, figsize=(16,6)):
  '''Function to calculates the % change, to transform it within a certain size of rolling window or or to plot it
  - Paramters:
    df: DataFrame or Series
    window: size of rolling window, 5 default
    show: is or not to display the rolling data, default True
    figsize: size the plot figure
  - Return value:
    df_rolling: serie
  '''
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Calculate % change based on rolling window
  df_rolling = df.rolling(window=window).apply(pc_last_previous)
  if show:
    # Plot the raw data and rolling data
    fig, axs = plt.subplots(1,2,figsize=figsize)
    ax = df.plot(ax=axs[0], title="Raw data", xlabel="Time", ylabel="Mean")
    ax = df_rolling.plot(ax=axs[1], title="Rolling data", xlabel="Time", ylabel="Mean")
    plt.tight_layout()
    plt.show()
  return df_rolling

def forecast_warning(df, level_1=None, level_2=None, level_3=None, window=3, report=True):
  '''Function to forecast the warning time
  - Paramters:
    df: DataFrame or Series
    level_1: fist warning threshold
    level_2: second warning threshold
    level_3: third warning threshold
    report: print summary report
  - Return value: None
  '''
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Check the frequency and resample if necessary
  if df.index.freqstr != 'H':
    df = df.resample('H').mean()
  # Initialize dataframe
  df['Level 1'] = None
  df['Level 2'] = None
  df['Level 3'] = None
  # Test for level 1
  df_rolling = df.rolling(window=window).mean().fillna(method='bfill')
  df.loc[df.iloc[np.where(df_rolling.iloc[:,0] >= level_1)].index, 'Level 1'] = 'Warning'
  df.loc[df.iloc[np.where(df_rolling.iloc[:,0] < level_1)].index, 'Level 1'] = 'Safe'
  # Test for level 2
  for ind in df.index[:-2]:
    ind_1 = ind
    ind_2 = ind+datetime.timedelta(hours=1)
    ind_3 = ind+datetime.timedelta(hours=2)
    if df.loc[ind_1].iloc[0] >= level_2 and df.loc[ind_2].iloc[0] >= level_2 and df.loc[ind_3].iloc[0] >= level_2:
      df.loc[ind_3, 'Level 2'] = 'Warning'
    if df.loc[ind_1, 'Level 2'] != 'Warning':
      df.loc[ind_1, 'Level 2'] = 'Safe'
    if df.loc[ind_2, 'Level 2'] != 'Warning':
      df.loc[ind_2, 'Level 2'] = 'Safe'
    if df.loc[ind_3, 'Level 2'] != 'Warning':
      df.loc[ind_3, 'Level 2'] = 'Safe'
  # Test for level 3
  df.loc[df.iloc[np.where(df.iloc[:,0] >= level_3)].index, 'Level 3'] = 'Warning'
  df.loc[df.iloc[np.where(df.iloc[:,0] < level_3)].index, 'Level 3'] = 'Safe'
  if report:
    print("\t\tSummary of Warning Forecasting Results\n", 
       "=========================================================================\n",
       "Level 1(AQI) - Hourly Rolling(window=3) Average Exceeds {}\n".format(level_1),
       "Level 2(AQI) - Hourly Average Exceeds {} for 3 consecutive hours\n".format(level_2),
       "Level 3(AQI) - Hourly Average Exceeds {}\n".format(level_3),
       "-------------------------------------------------------------------------\n"
       " Results for Level_1\n", 
       "=========================================================================\n"
       )
    result_level_1 = df.iloc[np.where(df['Level 1'] == 'Warning')].iloc[:,0]
    for time, value in zip(result_level_1.index, result_level_1.values):
      print(" Time: {}\t".format(time), "Value: {}".format(value))
    print("\n Results for Level_2\n",
      "=========================================================================")
    result_level_2 = df.iloc[np.where(df['Level 2'] == 'Warning')].iloc[:,0]
    for time, value in zip(result_level_2.index, result_level_2.values):
      print(" Time: {}\t".format(time), "Value: {}".format(value))
    print("\n Results for Level_3\n", 
      "=========================================================================")
    result_level_3 = df.iloc[np.where(df['Level 3'] == 'Warning')].iloc[:,0]
    for time, value in zip(result_level_3.index, result_level_3.values):
      print(" Time: {}\t".format(time), "Value: {}".format(value))
    print("=========================================================================")
  return


def nomalize_data(df=None):
  """Function to normalize the time series
  - Paramters:
    df: DataFrame or Series
  - Return value:
    df: DataFrame normalized
    avgs: average of each column, used to convert the prediction result
    devs: deviation of each column, used to convert the prediction result
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  avgs = df.mean()
  devs = df.std()
  for col in df.columns:
    df[col] = (df[col] - avgs.loc[col]) / devs.loc[col]
  return df, avgs, devs

def remove_volatility_seasonality(df=None):
  """Function to remove volatility and easonality of time series
  - Paramters:
    df: DataFrame or Series
  - Return value:
    df: DataFrame with volatility removed
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Get all the column names
  column_names = df.columns.values
  # Remove volatility
  daily_volatility = df.groupby(df.index.day).std()
  for i, column_name in enumerate(column_names):
    df[column_name + '_daily_vol'] = df.index.map(lambda d: daily_volatility.loc[d.day, column_name])
    df[column_name] = df[column_name] / df[column_name + '_daily_vol']
  # Remove seasonality
  daily_avgs = df.groupby(df.index.day).mean()
  for i, column_name in enumerate(column_names):
    df[column_name + '_daily_avg'] = df.index.map(lambda d: daily_avgs.loc[d.day, column_name])
    df[column_name] = df[column_name] - df[column_name + '_daily_avg']
  return df

def Reconvert_prediction(df_to_convert=None, df_trans=None, avgs=None, devs=None):
  """Function to reconvert the prediction result
  - Paramters:
    df_to_convert: DataFrame or Series, prediction result
    df_trans: DataFrame or Series, time series with volatility and easonality removed
    avgs: average of each column
    devs: deviation of each column
  - Return value:
    df: DataFrame reconverted
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df_to_convert, pd.DataFrame):
    df_to_convert = df_to_convert.to_frame()
  if not isinstance(df_trans, pd.DataFrame):
    df_trans = df_trans.to_frame()
  # Get all the column names
  column_names = df_to_convert.columns.values
  for col in column_names:
    df_to_convert[col] = (df_to_convert[col] + df_trans[col + '_daily_avg']) * df_trans[col + '_daily_vol']
    df_to_convert[col] = df_to_convert[col] * devs.loc[col] + avgs.loc[col]
  return df_to_convert
# =============================*END - General functions*=============================


# =============================*BLOC - SARIMAX Model functions*=============================
def grid_search_AIC_sarimax(df=None, exog=None, p=range(0,2), d=range(0,2), q=(0,2), P=range(0,3), D=range(0,3), Q=range(0,3), seasonal_lags=None, return_aic=False):
  """Function to determine the best orders based on the possible values we found by ACF and PACF plot
  - Paramters:
    df: DataFrame or Series
    exog: Array of exogenous regressors
    p: range of p in list, default (0,2)
    d: range of d in list, default (0,2)
    q: range of q in list, default (0,2)
    P: range of P in list, default (0,3)
    D: range of D in list, default (0,3)
    Q: range of Q in list, default (0,3)
    seasonal_lags: seasonal lags of data
    return_aic: is or not to return AIC evaluation dataframe
  - Return value:
    pdq: trend orders
    seasonal_pdq: seasonal orders
    order_aic: AIC search result
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  p, d, q, P, D, Q = p, d, q, P, D, Q
  # Get all the possible combination of orders
  pdq = list(itertools.product(p, d, q))
  seasonal_pdq = [(x[0], x[1], x[2], seasonal_lags) for x in list(itertools.product(P, D, Q))]
  # Initialize the AIC evaluation dict
  order_aic = {}
  for param in pdq:
    for param_seasonal in seasonal_pdq:
      try:
        model = SARIMAX(df, exog=exog, order=param, seasonal_order=param_seasonal)
        results = model.fit()
        order_aic['Order'] = [param]
        order_aic['Seasonal order'] = [param_seasonal]
        order_aic['AIC'] = [results.aic]
      except:
        continue
  # Transform AIC evaluation dict to dataframe
  order_aic = pd.DataFrame.from_dict(order_aic)
  # Affect the column name
  order_aic.columns=['Order', 'Seasonal order', 'AIC']
  # Get the minimum AIC tuple
  order_aic_min = order_aic.loc[np.where(order_aic['AIC'] == order_aic['AIC'].min())]
  # Get the best orders and seasonal orders
  pdq = order_aic_min.iloc[0]['Order']
  seasonal_pdq = order_aic_min.iloc[0]['Seasonal order']
  if return_aic:
    return pdq, seasonal_pdq, order_aic
  else:
    return pdq, seasonal_pdq


def fit_sarimax(df=None, exog=None, order=None, seasonal_order=None, trend=None, figsize=(10, 4), enforce_stationarity=True, enforce_invertibility=True):
  """Function of fitting SARIMAX to data
  - Paramters:
    df: DataFrame or Series
    exog: Array of exogenous regressors
    order: trend orders
    seasonal_order: seasonal orders
    trend: Parameter controlling the deterministic trend polynomial, {‘n’,’c’,’t’,’ct’} 
    figsize: size the plot figure
  - Return value:
    fitted model
  """
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  # Fit SARIMAX model
  model = SARIMAX(df, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility).fit()
  print(model.summary())
  model.plot_diagnostics(figsize=figsize)
  plt.tight_layout()
  plt.show()
  return model


def residual_check_sarimax(df=None, lags=None, figsize=(8,2)):
  """Function to use Ljung Box test to inspect if the residuals are correlated
  - Parameters:
    df: DataFrame or Series
    lags: lags to take
    figsize: size of figure
  - Return value: None
  """
  # Perform the Ljung-Box test
  lb_test = acorr_ljungbox(df, lags=lags)
  p_value = pd.DataFrame(lb_test[1])
  # Plot the p-values
  fig, ax = plt.subplots(figsize=figsize)
  ax.plot(p_value, linewidth=3, linestyle='--', label="Residual std={:.2f}".format(df.std()))
  ax.set_xlabel('Lags', fontsize=12)
  ax.set_ylabel('P-value', fontsize=12)
  ax.set_title('Ljung Box test - residual autocorrelation', fontsize=15)
  ax.legend(loc='best')
  plt.show()
  return


def predict_plot_sarimax(model=None, df_original=None, start=None, df_to_predict=None, exog=None, return_predict=False, return_rmse=False, zoom=False, zoom_start_at=None, zoom_end_at=None, figsize=(10, 2)):
  """Function to make predictions on testing set and visualize the result at one time, for the case no nomalization on the data
  - Parameters:
    model: fitted model
    df_original: DataFrame or Series of original data
    start: Int, str, or datetime, set for in-sample forecast
    df_to_predict: DataFrame or Series to predict, set for out-of-sample forceast
    plot: is or not to plot both orgianal data and prediction result
    return_predict: is or not to return predict value in Dataframe
    return_rmse:  is or not to return rmse evalution
    zoom: zoom the prediction result for a selected period
    zoom_start_at: str or datetime, start time point to zoom
    zoom_end_at: str or datetime, end time point to zoom
    figsize: size of figure, (10,2) default
  - Return value:
    predicted_values: result of predictions
    rmse: Root Mean Square Error of prediction
  """
  fig, ax = plt.subplots(figsize=figsize)
  if start is not None:
      # One-step-ahead prediction 
    forecast_osa_in = model.get_prediction(start=start, exog=exog)
    mean_forecast_osa_in = forecast_osa_in.predicted_mean.to_frame()
    mean_forecast_osa_in.columns = [df_original.columns.values[0]]
    osa_ci_in = forecast_osa_in.conf_int()
      # Dynamic Prediction 
    forecast_dyn_in = model.get_prediction(start=start, exog=exog, dynamic=True)
    mean_forecast_dyn_in = forecast_dyn_in.predicted_mean
    mean_forecast_dyn_in.columns = [df_original.columns.values[0]]
    dyn_ci_in = forecast_dyn_in.conf_int()
      # Calculate the Root Mean Square Error of prediction
    rmse_osa_in = np.sqrt(np.mean(np.square(df_original[start:].values - mean_forecast_osa_in.values)))
    rmse_dyn_in = np.sqrt(np.mean(np.square(df_original[start:].values - mean_forecast_dyn_in.values)))
      # Visualize the predcitions result
    if df_original is not None:
        # Transform Series objet to DataFrame object
      if not isinstance(df_original, pd.DataFrame):
        df_original = df_original.to_frame()
      ax.plot(df_original[start:], linewidth=3, label='observation')
    ax.plot(mean_forecast_osa_in.index, mean_forecast_osa_in.values, 'r-+', linewidth=3, label="One-step-ahead forecast (RMSE={:0.2f})".format(rmse_osa_in))
    ax.fill_between(osa_ci_in.index, osa_ci_in.iloc[:,0], osa_ci_in.iloc[:,1], color='r', alpha=0.05)
    ax.plot(mean_forecast_dyn_in.index, mean_forecast_dyn_in.values,  'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_in))
    ax.fill_between(dyn_ci_in.index, dyn_ci_in.iloc[:,0], dyn_ci_in.iloc[:,1], color='g', alpha=0.05)
    ax.legend(loc='best', fontsize=9)
    ax.set_title("In-sampling prediction", fontsize=22, fontweight="bold")
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(str(df_original.columns.values[0])+'-AQI', fontsize=18)
     # Zoom the prediction result for a indicated period
    if zoom:
      fig, ax_zoom = plt.subplots(figsize=figsize)
      zoom_start_at = pd.to_datetime(zoom_start_at)
      zoom_end_at = pd.to_datetime(zoom_end_at)
      if df_original is not None:
        # Transform Series objet to DataFrame object
        if not isinstance(df_original, pd.DataFrame):
          df_original = df_original.to_frame()
        df_original = df_original.loc[zoom_start_at:zoom_end_at]
        ax_zoom.plot(df_original, linewidth=3, label='observation')
        # Transform datetime string to Timestampe
      df_forecast_osa_in = mean_forecast_dyn_in.loc[zoom_start_at:zoom_end_at]
      df_forecast_dyn_in = mean_forecast_dyn_in.loc[zoom_start_at:zoom_end_at]
      rmse_osa_in = np.sqrt(np.mean(np.square(df_original.values - mean_forecast_dyn_in.values)))
      rmse_dyn_in = np.sqrt(np.mean(np.square(df_original.values - mean_forecast_dyn_in.values)))
      ax_zoom.plot(df_forecast_osa_in.index, df_forecast_osa_in.values, 'r-+', linewidth=3, label="One-step-ahead forecast (RMSE={:0.2f})".format(rmse_osa_in))
      ax_zoom.plot(df_forecast_dyn_in.index, df_forecast_dyn_in.values, 'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_in))
      ax_zoom.legend(loc='best', fontsize=9)
      ax_zoom.set_title("Zoom prediction from " + str(zoom_start_at) + " to " + str(zoom_end_at), fontsize=14, fontweight="bold")
      ax_zoom.set_xlabel('Time', fontsize=18)
      ax_zoom.set_ylabel(df_forecast_dyn_in.columns.values[0]+'-AQI', fontsize=18)
      for index in df_forecast_dyn_in.index:
        ax_zoom.axvline(index, linestyle='--', color='k', alpha=0.2)
      ax.axvspan(zoom_start_at, zoom_end_at, color='red', alpha=0.05)
    plt.show()
    if return_predict == True and return_rmse==True:
      return mean_forecast_osa_in, mean_forecast_dyn_in, rmse_osa_in, rmse_dyn_in
    if return_predict == True and return_rmse==False:
      return mean_forecast_osa_in, mean_forecast_dyn_in
    if return_predict == False and return_rmse==False:
      return
    # Out-of-sample prediction and confidence bounds
  if df_to_predict is not None:
    if not isinstance(df_to_predict, pd.DataFrame):
      df_to_predict = df_to_predict.to_frame()
      # One-step-ahead prediction
    forecast_osa_out = model.get_prediction(start=df_to_predict.index[0],end=df_to_predict.index[-1], exog=exog)
    mean_forecast_osa_out = forecast_osa_out.predicted_mean.to_frame()
    mean_forecast_osa_out.columns = [df_to_predict.columns.values[0]]
    osa_ci_out = forecast_osa_out.conf_int()
      # Dynamic Prediction
    forecast_dyn_out = model.get_prediction(start=df_to_predict.index[0],end=df_to_predict.index[-1], exog=exog, dynamic=True)
    mean_forecast_dyn_out = forecast_dyn_out.predicted_mean.to_frame()
    mean_forecast_dyn_out.columns = [df_to_predict.columns.values[0]]
    dyn_ci_out = forecast_dyn_out.conf_int()
      # Calculate the Root Mean Square Error of prediction
    rmse_osa_out = np.sqrt(np.mean(np.square(df_to_predict.values - mean_forecast_osa_out.values)))
    rmse_dyn_out = np.sqrt(np.mean(np.square(df_to_predict.values - mean_forecast_dyn_out.values)))
      # Visualize the predcitions result
    if df_original is not None:
      # Transform Series objet to DataFrame object
      if not isinstance(df_original, pd.DataFrame):
        df_original = df_original.to_frame()
      ax.plot(df_original[df_to_predict.index[0]:df_to_predict.index[-1]], linewidth=3, label='observation')
    ax.plot(mean_forecast_osa_out.index, mean_forecast_osa_out.values, 'r-+', linewidth=3, label="One-step-ahead forecast (RMSE={:0.2f})".format(rmse_osa_out))
    ax.fill_between(osa_ci_out.index, osa_ci_out.iloc[:,0], osa_ci_out.iloc[:,1], color='r', alpha=0.05)
    ax.plot(mean_forecast_dyn_out.index, mean_forecast_dyn_out.values,  'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_out))
    ax.fill_between(dyn_ci_out.index, dyn_ci_out.iloc[:,0], dyn_ci_out.iloc[:,1], color='g', alpha=0.05)
    ax.legend(loc='best', fontsize=9)
    ax.set_title("Out-of-sampling prediction", fontsize=22, fontweight="bold")
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(str(df_to_predict.columns.values[0])+'-AQI', fontsize=18)
    plt.show()
    # Zoom the prediction result for a indicated period
    if zoom:
      fig, ax_zoom = plt.subplots(figsize=figsize)
      # Transform datetime string to Timestampe
      zoom_start_at = pd.to_datetime(zoom_start_at)
      zoom_end_at = pd.to_datetime(zoom_end_at)
      if df_original is not None:
        # Transform Series objet to DataFrame object
        if not isinstance(df_original, pd.DataFrame):
          df_original = df_original.to_frame()
        df_original = df_original.loc[zoom_start_at:zoom_end_at]
        ax_zoom.plot(df_original, linewidth=3, label='observation')
      df_forecast_osa_out = mean_forecast_osa_out.loc[zoom_start_at:zoom_end_at]
      df_forecast_dyn_out = mean_forecast_dyn_out.loc[zoom_start_at:zoom_end_at]
      rmse_osa_out = np.sqrt(np.mean(np.square(df_original.values - df_forecast_osa_out.values)))
      rmse_dyn_out = np.sqrt(np.mean(np.square(df_original.values - df_forecast_dyn_out.values)))
      ax_zoom.plot(df_forecast_osa_out.index, df_forecast_osa_out.values, 'r-+', linewidth=3, label="One-step-ahead forecast (RMSE={:0.2f})".format(rmse_osa_out))
      ax_zoom.plot(df_forecast_dyn_out.index, df_forecast_dyn_out.values, 'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_out))
      ax_zoom.legend(loc='best', fontsize=9)
      ax_zoom.set_title("Zoom prediction from " + str(zoom_start_at) + " to " + str(zoom_end_at), fontsize=14, fontweight="bold")
      ax_zoom.set_xlabel('Time', fontsize=18)
      ax_zoom.set_ylabel(df_forecast_dyn_out.columns.values[0]+'-AQI', fontsize=18)
      for index in df_forecast_dyn_out.index:
        ax_zoom.axvline(index, linestyle='--', color='k', alpha=0.2)
      ax.axvspan(zoom_start_at, zoom_end_at, color='red', alpha=0.05)
    if return_predict == True and return_rmse==True:
      return mean_forecast_osa_out, mean_forecast_dyn_out, rmse_osa_out, rmse_dyn_out
    if return_predict == True and return_rmse==False:
      return mean_forecast_osa_out, mean_forecast_dyn_out
    if return_predict == False and return_rmse==False:
      return


def predict_sarimax(model=None, start_at_in_sampling=None, start_at_out_sampling=None, end_at_out_sampling=None, exog=None):
  """Function to make predictions on testing set and return prediction result(do not return the confidence interval)
  - Parameters:
    model: fitted model
    start_at_in_sampling: Int, str, or datetime, set for in-sampling forecast
    start_at_out_sampling: str or datetime, set for out-sampling forecast
    end_at_out_sampling: str or datetime, set for out-sampling forecast
    exog: Series or DataFrame, exogenous variable
  - Return value:
    predicted_values: result of predictions
  """
  if start_at_in_sampling is not None: # In_sampling forecast
      # One-step-ahead prediction 
    forecast_osa_in = model.get_prediction(start=start_at_in_sampling, exog=exog)
    mean_forecast_osa_in = forecast_osa_in.predicted_mean.to_frame()
    # osa_ci_in = forecast_osa_in.conf_int() # Get the confidence interval
      # Dynamic Prediction 
    forecast_dyn_in = model.get_prediction(start=start_at_in_sampling, exog=exog, dynamic=True)
    mean_forecast_dyn_in = forecast_dyn_in.predicted_mean.to_frame()
    # dyn_ci_in = forecast_dyn_in.conf_int()
    # return mean_forecast_osa_in, osa_ci_in, mean_forecast_dyn_in, dyn_ci_in
    return mean_forecast_osa_in, mean_forecast_dyn_in
  if start_at_out_sampling is not None and end_at_out_sampling is not None:
    start_at_out_sampling = pd.to_datetime(start_at_out_sampling)
    end_at_out_sampling = pd.to_datetime(end_at_out_sampling)
      # Dynamic Prediction
    forecast_dyn_out = model.get_prediction(start=start_at_out_sampling, end=end_at_out_sampling, exog=exog, dynamic=True)
    mean_forecast_dyn_out = forecast_dyn_out.predicted_mean.to_frame()
    # dyn_ci_out = forecast_dyn_out.conf_int()
    # return mean_forecast_dyn_out, dyn_ci_out
    return mean_forecast_dyn_out


def prediction_plot_sarimax(df_original=None, df_forecast=None, figsize=(10,2), zoom=False, zoom_start_at=None, zoom_end_at=None):
    '''
    Function to visualize the prediction result
    - Parameters:
      df_original: DataFrame or Series of original data
      df_forecast: DataFrame or Series to predict, set for out-of-sample forceast
      figsize: size of figure, (12,4) default
      zoom: zoom the prediction result for a selected period
      zoom_start_at: str or datetime, start time point to zoom
      zoom_end_at: str or datetime, end time point to zoom
    - Return value: None
    '''
    fig, ax = plt.subplots(figsize=figsize)
     # In-sample prediction and confidence bounds
    if len(df_forecast) == 2:
      df_forecast_osa_in = df_forecast[0]
      df_forecast_dyn_in = df_forecast[1]
        # Transform Series objet to DataFrame object
      if not isinstance(df_forecast_osa_in, pd.DataFrame):
        df_forecast_osa_in = df_forecast_osa_in.to_frame()
      if not isinstance(df_forecast_dyn_in, pd.DataFrame):
        df_forecast_dyn_in = df_forecast_dyn_in.to_frame()
      if df_original is not None:
          # Transform Series objet to DataFrame object
        if not isinstance(df_original, pd.DataFrame):
          df_original = df_original.to_frame()
          # Calculate the Root Mean Square Error of prediction
        rmse_osa_in = np.sqrt(np.mean(np.square(df_original.values - df_forecast_osa_in.values)))
        rmse_dyn_in = np.sqrt(np.mean(np.square(df_original.values - df_forecast_dyn_in.values)))
        ax.plot(df_original, linewidth=3, label='observation')
        ax.plot(df_forecast_osa_in.index, df_forecast_osa_in.iloc[:,0].values, 'r-+', linewidth=3, label="One-step-ahead forecast (RMSE={:0.2f})".format(rmse_osa_in))
        ax.plot(df_forecast_dyn_in.index, df_forecast_dyn_in.iloc[:,0].values,  'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_in))
      else:
        ax.plot(df_forecast_osa_in.index, df_forecast_osa_in.iloc[:,0].values, 'r-+', linewidth=3, label="One-step-ahead forecast")
        ax.plot(df_forecast_dyn_in.index, df_forecast_dyn_in.iloc[:,0].values,  'g--', linewidth=3, label="Dynamic forecast")
      # ax.fill_between(df_forecast_osa_in.index, df_forecast_osa_in.iloc[:,1], df_forecast_osa_in.iloc[:,2], color='r', alpha=0.05)
      # ax.fill_between(df_forecast_dyn_in.index, df_forecast_dyn_in.iloc[:,1], df_forecast_dyn_in.iloc[:,2], color='g', alpha=0.05)
      ax.legend(loc='best', fontsize=9)
      ax.set_title("In-sampling prediction", fontsize=22, fontweight="bold")
      ax.set_xlabel('Time', fontsize=18)
      ax.set_ylabel(str(df_forecast_osa_in.columns.values[0])+'-AQI', fontsize=18)
      # Zoom the prediction result for a indicated period
      if zoom:
        fig, ax_zoom = plt.subplots(figsize=figsize)
          # Transform datetime string to Timestampe
        zoom_start_at = pd.to_datetime(zoom_start_at)
        zoom_end_at = pd.to_datetime(zoom_end_at)
        df_forecast_osa_in = df_forecast_osa_in.loc[zoom_start_at:zoom_end_at]
        df_forecast_dyn_in = df_forecast_dyn_in.loc[zoom_start_at:zoom_end_at]
        if df_original is not None:
          # Transform Series objet to DataFrame object
          if not isinstance(df_original, pd.DataFrame):
            df_original = df_original.to_frame()
          df_original = df_original.loc[zoom_start_at:zoom_end_at]
          # Calculate the Root Mean Square Error of prediction
          rmse_osa_in = np.sqrt(np.mean(np.square(df_original.values - df_forecast_osa_in.values)))
          rmse_dyn_in = np.sqrt(np.mean(np.square(df_original.values - df_forecast_dyn_in.values)))
          ax_zoom.plot(df_original, linewidth=3, label='observation')
          ax_zoom.plot(df_forecast_osa_in.index, df_forecast_osa_in.iloc[:,0].values, 'r-+', linewidth=3, label="One-step-ahead forecast (RMSE={:0.2f})".format(rmse_osa_in))
          ax_zoom.plot(df_forecast_dyn_in.index, df_forecast_dyn_in.iloc[:,0].values,  'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_in))
        else:
          ax_zoom.plot(df_forecast_osa_in.index, df_forecast_osa_in.iloc[:,0].values, 'r-+', linewidth=3, label="One-step-ahead forecast")
          ax_zoom.plot(df_forecast_dyn_in.index, df_forecast_dyn_in.iloc[:,0].values,  'g--', linewidth=3, label="Dynamic forecast")
        ax_zoom.legend(loc='best', fontsize=9)
        ax_zoom.set_title("Zoom prediction from " + str(zoom_start_at) + " to " + str(zoom_end_at), fontsize=14, fontweight="bold")
        ax_zoom.set_xlabel('Time', fontsize=18)
        ax_zoom.set_ylabel(df_forecast_dyn_in.columns.values[0]+'-AQI', fontsize=18)
        for index in df_forecast_dyn_in.index:
          ax_zoom.axvline(index, linestyle='--', color='k', alpha=0.2)
        ax.axvspan(zoom_start_at, zoom_end_at, color='red', alpha=0.05)
    else:
      df_forecast_dyn_out = df_forecast
        # Transform Series objet to DataFrame object
      if not isinstance(df_forecast_dyn_out, pd.DataFrame):
        df_forecast_dyn_out = df_forecast_dyn_out.to_frame()
      if df_original is not None:
          # Transform Series objet to DataFrame object
        if not isinstance(df_original, pd.DataFrame):
          df_original = df_original.to_frame()
        rmse_dyn_out = np.sqrt(np.mean(np.square(df_original.values - df_forecast_dyn_out.values)))
        ax.plot(df_original, linewidth=3, label='observation')
        ax.plot(df_forecast_dyn_out.index, df_forecast_dyn_out.values,  'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_out))
      else:
        ax.plot(df_forecast_dyn_out.index, df_forecast_dyn_out.values,  'g--', linewidth=3,label="Dynamic forecast")
      # ax.fill_between(df_forecast_dyn_out.index, df_forecast_dyn_out.iloc[:,1], df_forecast_dyn_out[:,2], color='g', alpha=0.05)
      ax.legend(loc='best', fontsize=9)
      ax.set_title("Out-of-sampling prediction", fontsize=22, fontweight="bold")
      ax.set_xlabel('Time', fontsize=18)
      ax.set_ylabel(str(df_forecast_dyn_out.columns.values[0])+'-AQI', fontsize=18)
      if zoom:
        fig, ax_zoom = plt.subplots(figsize=figsize)
          # Transform datetime string to Timestampe
        zoom_start_at = pd.to_datetime(zoom_start_at)
        zoom_end_at = pd.to_datetime(zoom_end_at)
        df_forecast_dyn_out = df_forecast_dyn_out.loc[zoom_start_at:zoom_end_at]
        if df_original is not None:
          # Transform Series objet to DataFrame object
          if not isinstance(df_original, pd.DataFrame):
            df_original = df_original.to_frame()
          df_original = df_original.loc[zoom_start_at:zoom_end_at]
          # Calculate the Root Mean Square Error of prediction
          rmse_dyn_out = np.sqrt(np.mean(np.square(df_original.values - df_forecast_dyn_out.values)))
          ax_zoom.plot(df_original, linewidth=3, label='observation')
          ax_zoom.plot(df_forecast_dyn_out.index, df_forecast_dyn_out.iloc[:,0].values,  'g--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse_dyn_out))
        else:
          ax_zoom.plot(df_forecast_dyn_out.index, df_forecast_dyn_out.iloc[:,0].values,  'g--', linewidth=3, label="Dynamic forecast")
        ax_zoom.legend(loc='best', fontsize=9)
        ax_zoom.set_title("Zoom prediction from " + str(zoom_start_at) + " to " + str(zoom_end_at), fontsize=14, fontweight="bold")
        ax_zoom.set_xlabel('Time', fontsize=18)
        ax_zoom.set_ylabel(df_forecast_dyn_out.columns.values[0]+'-AQI', fontsize=18)
        for index in df_forecast_dyn_out.index:
          ax_zoom.axvline(index, linestyle='--', color='k', alpha=0.2)
        ax.axvspan(zoom_start_at, zoom_end_at, color='red', alpha=0.05)
    plt.show()
    return
# =============================*END - SARIMAX Model functions*=============================


# =============================*BLOC - VAR Model functions*=============================
def grangers_causation_matrix(df=None, test='ssr_chi2test', maxlag=12, verbose=False):    
    """Funtion to Check Granger Causality of all possible combinations of the time series
    - Parameters:
      df: Dataframe
      maxlag: the Granger causality test results are calculated for all lags up to maxlag
      verbose: print results if true
    - Return:
      df_p_value: test P-value matrix
    - Remark:
      The rows are the response variable, columns are predictors. The values in the table 
      are the P-values. P-values lesser than the significance level (0.05), implies 
      the Null Hypothesis that the coefficients of the corresponding past values is 
      zero, that is, the X does not cause Y can be rejected.
    """
    # Transform Series objet to DataFrame object
    if not isinstance(df, pd.DataFrame):
      df = df.to_frame()
    df_p_value = pd.DataFrame(np.zeros((len(df.columns), len(df.columns))), columns=df.columns, index=df.columns)
    for column in df.columns:
        for row in df.columns:
            test_result = grangercausalitytests(df[[row, column]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: 
              print(f'Y = {row}, X = {column}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df_p_value.loc[row, column] = min_p_value
    df_p_value.columns = [var + '_x' for var in df.columns]
    df_p_value.index = [var + '_y' for var in df.columns]
    return df_p_value


def cointegration_test(df=None, det_order=-1, k_ar_diff=5, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary
    - Parameters:
      df: DataFrame or Series
      det_order: ['-1 - no deterministic terms', '0 - constant term', '1 - linear trend']
      k_ar_diff: Number of lagged differences in the model
      alpha:  significance level
    - Return value: None
    """
    # Transform Series objet to DataFrame object
    if not isinstance(df, pd.DataFrame):
      df = df.to_frame()
    result = coint_johansen(endog=df, det_order=det_order, k_ar_diff=k_ar_diff )
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = result.lr1 # Trace statistic
    cvts = result.cvt[:, d[str(1-alpha)]] # Critical values (90%, 95%, 99%) for trace statistic
    def adjust(val, length= 6): 
      return str(val).ljust(length)
    # Prit the summary result
    print('Name   |  Test Statistic | Critical Value({}%) | Cointegrated  \n'.format(int(100*(1-alpha))), '--'*30)
    for column, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(column), '| ', adjust(round(trace,2), 14), "|", adjust(cvt, 19), '|' , trace > cvt)


def prediction_plot_var(df_original=None, df_forecast=None, figsize=(10,2), zoom=False, zoom_start_at=None, zoom_end_at=None, title="VAR out-of-sampling prediction"):
  """Function to make predictions on testing set
  - Parameters:
    df_original: DataFrame or Series of original data
    df_forecast: DataFrame or Series to predict, out-of-sample forceast result
    figsize: size of figure, (10,2) default
    zoom: zoom the prediction result for a selected period
    zoom_start_at: str or datetime, start time point to zoom
    zoom_end_at: str or datetime, end time point to zoom
  - Return value: None
  """
  # Transform datetime string to Timestampe
  zoom_start_at = pd.to_datetime(zoom_start_at)
  zoom_end_at = pd.to_datetime(zoom_end_at)
  # Transform Series objet to DataFrame object
  if not isinstance(df_original, pd.DataFrame):
    df_original = df_original.to_frame()
  if not isinstance(df_forecast, pd.DataFrame):
    df_forecast = df_forecast.to_frame()
  # Calculate the Root Mean Square Error of prediction
  rmse = np.sqrt(np.mean(np.square(df_original.values - df_forecast.values)))
  fig, ax = plt.subplots(figsize=figsize)
  if df_original is not None:
    ax.plot(df_original, linewidth=3, label='observation')
  ax.plot(df_forecast.index, df_forecast.values, 'r--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse))
  ax.legend(loc='best', fontsize=9)
  ax.set_title(title, fontsize=22, fontweight="bold")
  ax.set_xlabel('Time', fontsize=18)
  ax.set_ylabel(df_original.columns.values[0]+'-AQI', fontsize=18)
  # Zoom the prediction result for a indicated period
  if zoom:
    df_original = df_original.loc[zoom_start_at:zoom_end_at]
    df_forecast = df_forecast.loc[zoom_start_at:zoom_end_at]
    fig, ax_zoom = plt.subplots(figsize=figsize)
    rmse = np.sqrt(np.mean(np.square(df_original.values - df_forecast.values)))
    if df_original is not None:
      ax_zoom.plot(df_original, linewidth=3, label='observation')
    ax_zoom.plot(df_forecast.index, df_forecast.values, 'r--', linewidth=3, label="Dynamic forecast (RMSE={:0.2f})".format(rmse))
    ax_zoom.legend(loc='best', fontsize=9)
    ax_zoom.set_title("Zoom prediction from " + str(zoom_start_at) + " to " + str(zoom_end_at), fontsize=14, fontweight="bold")
    ax_zoom.set_xlabel('Time', fontsize=18)
    ax_zoom.set_ylabel(df_original.columns.values[0]+'-AQI', fontsize=18)
    for index in df_forecast.index:
      ax_zoom.axvline(index, linestyle='--', color='k', alpha=0.2)
    ax.axvspan(zoom_start_at, zoom_end_at, color='red', alpha=0.05)
  return
# =============================*END - VAR Model functions*=============================


# =============================*BLOC - RNN Model functions*=============================
def preparation_data_rnn(df=None, lags=None):
  """Function to preprare taining and testing set for RNN model based on seasonal lags
  - Parameters:
    df: DataFrame or Series
    lags: seasonal lags
  - Retrun value:
    X: 1D array of inputing data
    y: 1D array of target value
  """
  X, y = [], []
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  for row in range(len(df) - lags):
  # Transform 2D data to 1D
    x_1d_array = []
    for i in zip(df.values[row:(row + lags)]):
      x_1d_array.append(i[0][0])
    X.append(x_1d_array)
    y_1d_array = []
    for i in zip(df.values[row + lags]):
      y_1d_array.append(i[0])
    y.append(y_1d_array)    
  return np.array(X), np.array(y)


def convert_to_dataframe_rnn(Y=None, y=None):
  """Function to convert Array into Dataframe
  - Parameters:
    Y: Original testing set based on which implementing the split (DataFrame)
    y: Array data to convert
  - Return value:
    y: Dataframe converted
  """
  # Transform Series objet to DataFrame object
  if not isinstance(Y, pd.DataFrame):
    Y = Y.to_frame()
  # Convert y to DataFrame
  y = pd.DataFrame(y)
  # Get the right index of y
  y.index = Y.index[len(Y)-len(y):]
  # Set the columns name of y
  y.columns = Y.columns 
  return y


# Fitting the RNN model
def fit_RNN(X=None, y=None, epochs=50, batch_size=None, verboses=2):
  '''Function to fit RNN model to data
  - Paramter:
    X: Input data, Array
    y: Target data, Array
    epochs: Number of epochs to train the model, default 50
    batch_size: Number of samples per gradient update
    verbose: Verbosity mode, verbose=2 is recommended when not running interactively
  - Return value:
    model: fitted model
  '''
  model = Sequential()
  model.add(Dense(4, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(x=X, y=y, epochs=epochs, batch_size=batch_size, verbose=verboses)
  return model


def predict_rnn(df_original=None, df_test=None, df_predict=None, figsize=(12, 4), plot=False, return_rmse=False, return_dataframe=False, zoom=False):
  '''Function to plot the prediction result
  - Parameters:
    df_original: DataFrame or Series of original data
    df_test: true value, Array or DataFrame
    df_predict: predicted value, Array or DataFrame
    figsize: size of plot figure
    plot: is or not to plot both orgianal data and prediction result
    return_rmse: is or not to return rmse of prediction
    zoom: is or not to zoom the prediction result part
  - Return value:
    return_rmse: Root Mean Square Error of prediction / df_test and df_predict(DataFrame)
  '''
  # Transform Series objet to DataFrame object
  if not isinstance(df_original, pd.DataFrame):
    df_original = df_original.to_frame()
  if not isinstance(df_test, pd.DataFrame):
    df_test = convert_to_dataframe_rnn(df_original, df_test)
  if not isinstance(df_predict, pd.DataFrame):
    df_predict = convert_to_dataframe_rnn(df_original, df_predict)
  # Calculate the Root Mean Square Error of prediction
  rmse = np.sqrt(np.mean(np.square(df_test.values - df_predict.values)))
  if plot:
    # Plot in-sample-prediction
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df_original, linewidth=3, label='Observation')
    ax.plot(df_test.index, df_test.values, linewidth=3, label='Truth')
    ax.plot(df_predict.index, df_predict.values, linewidth=3, label="prediction (RMSE={:0.2f})".format(rmse))
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(str(df_test.columns.values[0])+'/ppb', fontsize=18)
    ax.set_title('Prediction RNN', fontsize=22, fontweight="bold")
    plt.legend(loc='upper left')
    plt.show()
  if zoom:
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(df_test.index, df_test.values, linewidth=3, label='Truth')
    ax1.plot(df_predict.index, df_predict.values, linewidth=3, label="prediction (RMSE={:0.2f})".format(rmse))
    ax1.set_xlabel('Time', fontsize=18)
    ax1.set_ylabel(str(df_test.columns.values[0])+'/ppb', fontsize=18)
    ax1.set_title('Prediction RNN', fontsize=22, fontweight="bold")
    plt.legend(loc='upper left')
    plt.show()
  if return_rmse == True and return_dataframe == False:
    return rmse
  if return_rmse == False and return_dataframe == True:
    return df_test, df_predict
  if return_rmse == True and return_dataframe == True:
    return rmse, df_test, df_predict
  if return_rmse == False and return_dataframe == False:
    return


def predict_future_rnn(df=None, model=None, start=None, end=None, lags=None, figsize=(10,4), return_result=False, plot=False, width=0.03, title='Predictions', min_max_distance=0.1):
  '''Function to predict by RNN model for a future new period
  - Paramters:
    df: DataFrame or Series
    model: fitted model
    start: start time point to display(String or Timestampe)
    end: end time point to display(String or Timestampe)
    lags: seasonal lags
    figsize: size of figure, default (10,4)
    return_result: is or not to return prections values
    plot: is or not to plot dashbord
    width: control the width between each bar, default 0.03
    title: set tilte of plot
    min_max_distance: used to modeify the position of MIN and MAX value
  - Return value:
    predict_data: prediction values
  '''
  # Transform Series objet to DataFrame object
  if not isinstance(df, pd.DataFrame):
    df = df.to_frame()
  df = df.resample('H').mean()
  # Transform datetime string to Timestampe
  start = pd.to_datetime(start)
  end = pd.to_datetime(end)
  length = int((end-start)/datetime.timedelta(1/24) + 1)
  # Find the index and value of 'lags' previous data
  lags_data, predict_data = [], []
  for j in range(lags):
    index = start - datetime.timedelta((lags-j)/24)
    value = df.loc[index]
    lags_data.append(value[0])
  predict_value = model.predict(np.array([lags_data]))
  lags_data.append(predict_value[0,0])
  predict_data.append(predict_value)
  # Get all the prediction value
  for i in range(length-1):
    temp = []
    for j in range(lags):
      temp.append(lags_data[len(lags_data) - (lags-j)])
    predict_temp = model.predict(np.array([temp]))
    lags_data.append(predict_temp[0,0])
    predict_data.append(predict_temp)
  predict_data = np.array(predict_data).reshape(length,1)
  # Convert y to DataFrame
  predict_data = pd.DataFrame(predict_data)
  # Get the right index of y
  predict_data.index = pd.date_range(start=start, end=end, freq='H')
  # Set the columns name of y
  predict_data.columns = df.columns.values
  # Display Predict value Dashbord
  if plot:
    Air_Situation_Dashbord(predict_data, width=width, title=title, min_max_distance=min_max_distance)
  if return_result:
    return predict_data
  else:
    return
# =============================*END - RNN Model functions*=============================