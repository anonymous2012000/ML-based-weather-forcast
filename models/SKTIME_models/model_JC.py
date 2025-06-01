import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.split import ExpandingWindowSplitter
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.tbats import TBATS
import sktime.classification.deep_learning as dl_clf
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os


def split_data(dataset, forecast_horizon):
    dataset = dataset.dropna()
    dataset_train, dataset_test = temporal_train_test_split(dataset, fh=forecast_horizon)
    dataset_train_features = dataset_train.loc[:,
                             ['Summary', 'Precip Type', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
                              'Visibility (km)', 'Pressure (millibars)']]
    dataset_train_features.index = dataset_train_features.index.to_series()  # OR to_period(freq="h")
    dataset_train_features = dataset_train_features.dropna()
    dataset_train_target = dataset_train['Temperature (C)']
    dataset_train_target.index = dataset_train_target.index.to_series()
    dataset_train_target = dataset_train_target.dropna()
    return dataset_train_features, dataset_train_target, dataset_test


def classification_model(dataset_train_target, dataset_train_features, dataset_test, forecast_horizon, scores):
    # Kept this in for now, not sure whether to delete
    dataset_test_features = dataset_test.loc[:,[
            'Summary',
            'Precip Type',
            'Humidity',
            'Wind Speed (km_h)',
            'Wind Bearing (degrees)',
            'Visibility (km)',
            'Pressure (millibars)'
        ]]

    # Drop target column(s) from training & test sets
    # Columns outlined here are optional targets. Likely one will be chosen per iteration.
    X_train = dataset_train_features.drop(columns=['Summary', 'Precip Type', 'Daily Summary'])
    Y_train = dataset_train_features['Summary', 'Precip Type', 'Daily Summary']

    X_test = dataset_test_features.drop(columns=['Summary', 'Precip Type', 'Daily Summary'])
    Y_test = dataset_test_features['Summary', 'Precip Type', 'Daily Summary']

    # One hot encode categorical columns
    # Target label will be exluded from this below. (E.g. if target is 'Precip Type' this wont be encoded.
    categorical_cols = []
    if 'Summary' in X_train.columns:
        categorical_cols.append('Summary')
    if 'Precip Type' in X_train.columns:
        categorical_cols.append('Precip Type')
    if 'Daily Summary' in X_train.columns:
        categorical_cols.append('Daily Summary')

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    X_test = X_test.reindex(columns=X_train.columns)

    # Label encode the target
    labelEncoder = LabelEncoder()
    Y_train = labelEncoder.fit_transform(Y_train)
    Y_test = labelEncoder.fit_transform(Y_test)

    # Build LSTM classifier
    lstm_clf = dl_clf.LSTMFCNClassifier(n_epochs=10)
    lstm_clf.fit(X_train, Y_train)

    # Prediction
    Y_predict = lstm_clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict, average='weighted')

    print(f"Accuracy score: {accuracy}")
    print(f"f1 score: {f1}")

    # Add scores to list
    scores = scores._append({
        'Model': 'Classification-LSTMFCN',
        'Mean Absolute Error': None,
        'Root Mean Squared Error': None,
        'Mean Absolute Percentage Error': None,
        'Accuracy': accuracy,
        'F1-Score': f1
    })

    # save model for use later
    joblib.dump(lstm_clf, 'LSTM_Classifier_Model')

    return scores


def regression_model(dataset_train_target, dataset_train_features, dataset_test, forecast_horizon, scores):
    # LSTM with sktime

    # Grid search to find the optimal hyperparameters
    lstmParameters = {'encoder_n_layers': [2]}
    cv = ExpandingWindowSplitter(fh=forecast_horizon,
                                 step_length=1)  # Attempted fix, didnt work

    modelLSTM = ForecastingGridSearchCV(
        forecaster=NeuralForecastLSTM(),
        param_grid=lstmParameters,
        cv=cv)
    modelLSTM.fit(y=dataset_train_target,
                  X=dataset_train_features,
                  fh=forecast_horizon)

    print("BEST PARAMETERS:\n", modelLSTM.best_params_)
    gridPredictionsLSTM = modelLSTM.predict(fh=forecast_horizon)
    print(gridPredictionsLSTM)

    # Alternative to grid search is to do it manually, look at dissertation code, TBATS section for this.

    # build the model
    modelLSTM = NeuralForecastLSTM()
    modelLSTM.fit(y=dataset_train_target,
                  X=dataset_train_features,
                  fh=forecast_horizon)

    # make predictions
    lstmPredictions = modelLSTM.predict(fh=forecast_horizon)

    # calculate scores based off predictions
    lstmMAE = mean_absolute_error(dataset_test['Temperature (C)'], lstmPredictions)
    lstmRMSE = mean_squared_error(dataset_test['Temperature (C)'], lstmPredictions, square_root=True)
    lstmMAPE = mean_absolute_percentage_error(dataset_test['Temperature (C)'], lstmPredictions)

    # save scores to panda dataframe
    scores = scores.append({'Model': 'LSTM',
                            'Mean Absolute Error': lstmMAE,
                            'Root Mean Squared Error': lstmRMSE,
                            'Mean Absolute Percentage Error': lstmMAPE},
                           ignore_index=True)
    # Print Scores
    print("SCORES:\n", scores)

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
    modelSARIMA.fit(y=dataset_train_target,
                    X=dataset_train_features,
                    fh=forecast_horizon)

    # checks on X features
    print(dataset_train_features.shape)  # Should have rows
    print(dataset_test.shape)  # Should have 24 rows
    print(dataset_train_target.shape)  # Should have rows

    # Possible fix
    # Prepare test features (exogenous variables for the forecast horizon)
    dataset_test_features = dataset_test.loc[:, ['Summary', 'Precip Type', 'Humidity', 'Wind Speed (km_h)',  # Changed to comply with data preprocessing
                                                 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']]
    dataset_test_features.index = dataset_test_features.index.to_period(freq="h")

    # End of possible FIX

    # make predictions and save prediction
    sarimaPredictions = modelSARIMA.predict(fh=forecast_horizon,
                                            X=dataset_test_features)

    # calculate scores based off predictions
    sarimaMAE = mean_absolute_error(dataset_test['Temperature (C)'], sarimaPredictions)
    sarimaRMSE = mean_squared_error(dataset_test['Temperature (C)'], sarimaPredictions, square_root=True)
    sarimaMAPE = mean_absolute_percentage_error(dataset_test['Temperature (C)'], sarimaPredictions)

    # save scores to panda dataframe
    scores = scores._append({'Model': 'SARIMA',
                             'Mean Absolute Error': sarimaMAE,
                             'Root Mean Squared Error': sarimaRMSE,
                             'Mean Absolute Percentage Error': sarimaMAPE},
                            ignore_index=True)

    # save model for use later
    joblib.dump(modelSARIMA, 'SARIMA_Model')

    """
    TBATS with sktime
    """

    # build the model
    modelTBATS = TBATS(use_box_cox=True,
                       box_cox_bounds=False,
                       use_trend=False,
                       use_damped_trend=True,
                       use_arma_errors=True,
                       show_warnings=False,
                       sp=24)
    modelTBATS.fit(y=dataset_train_target,
                   X=dataset_train_features,
                   fh=forecast_horizon)

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
                             'Mean Absolute Percentage Error': tbatsMAPE},
                            ignore_index=True)

    # save model for use later
    joblib.dump(modelTBATS, 'TBATS_Model')
    return scores


# Determines which model is run based on data at hand
def model_selection(target_variable, dataset_train_features, dataset_train_target, dataset_test, forecast_horizon, scores):
    """
    Chooses regression if we're predicting values like temperature, rainfall etc.,
    Classification if we're predicting Summary, precipitation type etc.
    """
    classification_variables = ['Summary', 'Precip Type', 'Daily Summary']
    regression_variables = ['']
    if target_variable in regression_variables:
        scores = regression_model(
            dataset_train_target=dataset_train_target,
            dataset_train_features=dataset_train_features,
            dataset_test=dataset_test,
            forecast_horizon=forecast_horizon,
            scores=scores
        )
    elif target_variable in classification_variables:
        scores = classification_model(
            dataset_train_target=dataset_train_target,
            dataset_train_features=dataset_train_features,
            dataset_test=dataset_test,
            forecast_horizon=forecast_horizon,
            scores=scores
        )
    else:
        raise ValueError(
            f"Unknown target: {target_variable}."
        )

    return scores


def plot_results(scores):
    plt.figure(figsize=(12, 6))
    plt.bar(scores['Model'], scores['Mean Absolute Error'])
    plt.title('MAE of different ML models')
    plt.xlabel('Models')
    plt.ylabel('MAE scores')
    plt.show()
    return


def main():
    # Load dataset & initialise forecast horizon
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Path of current file
    csv_path = os.path.join(current_dir, '..', '..', 'CSV_Files', 'weatherHistory_p.csv')  # This can be easily amended to consider all submitted data files
    dataset = pd.read_csv(csv_path)
    forecast_horizon = ForecastingHorizon(np.arange(1, 25))

    # Define scoring metrics
    scores = pd.DataFrame(columns=[
        'Model',
        'Mean Absolute Error',
        'Root Mean Squared Error',
        'Mean Absolute Percentage Error'
    ])

    # Split data into training & test sets
    dataset_train_features, dataset_train_target, dataset_test = split_data(dataset, forecast_horizon)

    scores = regression_model(
        dataset_train_target,
        dataset_train_target,
        dataset_test,
        forecast_horizon=forecast_horizon,
        scores=scores
    )

    # Plot results using matplotlib
    plot_results(scores)


if __name__ == "__main__":
    main()
