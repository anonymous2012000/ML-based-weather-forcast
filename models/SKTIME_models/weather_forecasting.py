# -*- coding: utf-8 -*-
"""Weather_Forecasting.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1siph6kdWrN6gW3ndA_VeRdV5sso-EWju

# Start

Import all libraries
"""

!pip install sktime
!pip install neuralforecast
!pip install pmdarima
!pip install tbats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.split import ExpandingWindowSplitter
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.tbats import TBATS
import joblib

"""Load the data"""

dataset = pd.read_csv('/content/sample_data/weatherHistory.csv')
dataset

"""Simple data pre-processing"""

# rename column
dataset = dataset.rename(columns={'Date': 'Datetime'})

# convert Date Format
dataset['Datetime'] = pd.to_datetime(dataset['Datetime'].astype(str).str.replace(r'\+02:00', '', regex=True), errors='coerce')

# drop Unnecessary Column ( "Loud Column has 0 values")
if 'Loud Cover' in dataset.columns:
  dataset.drop(columns=['Loud Cover'], inplace=True)

# drop daily summary column as it has overlapping data
if 'Daily Summary' in dataset.columns:
  dataset.drop(columns=['Daily Summary'], inplace=True)

# convert categorical valus into numerical values
leSummary = LabelEncoder()
leSummary.fit(dataset['Summary'])
replaceSummary = leSummary.transform(dataset['Summary'])
dataset['Summary'] = replaceSummary
lePrecipType = LabelEncoder()
lePrecipType.fit(dataset['Precip Type'])
replacePrecipType = lePrecipType.transform(dataset['Precip Type'])
dataset['Precip Type'] = replacePrecipType

# duplciate datatime values need to be dropped
dataset = dataset.drop_duplicates(subset=['Datetime'])
# set the datatime as the index value
dataset = dataset.set_index('Datetime').asfreq('h')

# normalise values using min max scaling
dataset = (dataset - dataset.mean()) / dataset.std()

dataset

"""Set our forecasting horizon"""

forecast_horizon = ForecastingHorizon(np.arange(1, 25))

"""A dataframe will be used to keep track of all ML scores"""

scores = pd.DataFrame(columns=['Model', 'Mean Absolute Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error'])

"""Split the dataset up into a test train set"""

dataset = dataset.dropna()
dataset_train, dataset_test = temporal_train_test_split(dataset, fh=forecast_horizon)
dataset_train_features = dataset_train.loc[:, ['Summary',	'Precip Type',	'Humidity',	'Wind Speed (km/h)',	'Wind Bearing (degrees)',	'Visibility (km)',	'Pressure (millibars)']]
dataset_train_features.index = dataset_train_features.index.to_period(freq="h")
dataset_train_features = dataset_train_features.dropna()
dataset_train_target = dataset_train['Temperature (C)']
dataset_train_target.index = dataset_train_target.index.to_period(freq="h")
dataset_train_target = dataset_train_target.dropna()

"""# Models

## LSTM with sktime

Grid search can be used to optimise the hyperparameters but it takes an extremely long time to do so.
<br>
Code:
<br>

use grid search to find the optimal hyperparameters
lstmParameters = {'encoder_n_layers' : [2]}
cv = ExpandingWindowSplitter(fh=forecast_horizon)

modelLSTM = ForecastingGridSearchCV(forecaster=NeuralForecastLSTM(), param_grid=lstmParameters, cv=cv)
modelLSTM.fit(y=dataset_train_target, X=dataset_train_features, fh=forecast_horizon)

print best parameter after tuning
print(modelLSTM.best_params_)
gridPredictionsLSTM = modelLSTM.predict(fh=forecast_horizon)
print(gridPredictionsLSTM)

Alternative to grid search is to do it manually, look at dissertation code, TBATS section for this.
"""

# build the model
modelLSTM = NeuralForecastLSTM()
modelLSTM.fit(y=dataset_train_target, X=dataset_train_features, fh=forecast_horizon)

# make predictions
lstmPredictions = modelLSTM.predict(fh=forecast_horizon)

# calculate scores based off predictions
lstmMAE = mean_absolute_error(dataset_test['Temperature (C)'], lstmPredictions)
lstmRMSE = mean_squared_error(dataset_test['Temperature (C)'], lstmPredictions, square_root=True)
lstmMAPE = mean_absolute_percentage_error(dataset_test['Temperature (C)'], lstmPredictions)

# save scores to panda dataframe
scores = scores._append({'Model': 'LSTM',
                         'Mean Absolute Error': lstmMAE,
                         'Root Mean Squared Error': lstmRMSE,
                         'Mean Absolute Percentage Error' : lstmMAPE},
                        ignore_index=True)
#Print Scores
print(scores)

# save model for use later
joblib.dump(modelLSTM, 'LSTM_Model')

"""## SARIMA with sktime

Auto ARIMA from sktime to automatically adjust parameters for the model but it takes way too long to use.

An "X" value is not passed in here as it caused errors. Model will be better if this can be fixed. Also model parameters need to be adjusted.
"""

# build the model
modelSARIMA = ARIMA(order=(1, 1, 0),
                    seasonal_order=(0, 1, 0, 12))

# fit the model
modelSARIMA.fit(y=dataset_train_target, X=dataset_train_features, fh=forecast_horizon)

#checks on X features
print(dataset_train_features.shape)  # Should have rows
print(dataset_test.shape)            # Should have 24 rows
print(dataset_train_target.shape)    # Should have rows

#Possible fix
# Prepare test features (exogenous variables for the forecast horizon)
dataset_test_features = dataset_test.loc[:, ['Summary', 'Precip Type', 'Humidity', 'Wind Speed (km/h)',
                                            'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']]
dataset_test_features.index = dataset_test_features.index.to_period(freq="h")

# End of possible FIX

# make predictions and save prediction
sarimaPredictions = modelSARIMA.predict(fh=forecast_horizon, X=dataset_test_features)

# calculate scores based off predictions
sarimaMAE = mean_absolute_error(dataset_test['Temperature (C)'], sarimaPredictions)
sarimaRMSE = mean_squared_error(dataset_test['Temperature (C)'], sarimaPredictions, square_root=True)
sarimaMAPE = mean_absolute_percentage_error(dataset_test['Temperature (C)'], sarimaPredictions)

# save scores to panda dataframe
scores = scores._append({'Model': 'SARIMA',
                         'Mean Absolute Error': sarimaMAE,
                         'Root Mean Squared Error': sarimaRMSE,
                         'Mean Absolute Percentage Error' : sarimaMAPE},
                        ignore_index=True)

# save model for use later
joblib.dump(modelSARIMA, 'SARIMA_Model')

"""## TBATS with sktime

Using the best
"""

# build the model
modelTBATS = TBATS(use_box_cox=True,
                   box_cox_bounds=False,
                   use_trend=False,
                   use_damped_trend=True,
                   use_arma_errors=True,
                   show_warnings=False,
                   sp=24)
modelTBATS.fit(y=dataset_train_target, X=dataset_train_features, fh=forecast_horizon)

# make predictions
tbatsPredictions = modelTBATS.predict(fh=forecast_horizon)

# calculate scores based off predictions
tbatsMAE = mean_absolute_error(dataset_test['Temperature (C)'], tbatsPredictions)
tbatsRMSE = mean_squared_error(dataset_test['Temperature (C)'], tbatsPredictions, square_root=True)
tbatsMAPE = mean_absolute_percentage_error(dataset_test['Temperature (C)'], tbatsPredictions)

# save scores to panda dataframe
scores = scores._append({'Model': 'TBATS',
                         'Mean Absolute Error': tbatsMAE,
                         'Root Mean Squared Error': tbatsRMSE,
                         'Mean Absolute Percentage Error' : tbatsMAPE},
                        ignore_index=True)

# save model for use later
joblib.dump(modelTBATS, 'TBATS_Model')

"""# Results

Print out all the scores
"""

print(scores)

"""Plot Results onto a graph"""

plt.figure(figsize=(12, 6))
plt.bar(scores['Model'], scores['Mean Absolute Error'])
plt.title('MAE of different ML models')
plt.xlabel('Models')
plt.ylabel('MAE scores')
plt.show()