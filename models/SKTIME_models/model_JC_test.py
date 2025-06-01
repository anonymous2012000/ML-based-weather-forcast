import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
from sktime.forecasting.tbats import TBATS
import sktime.classification.deep_learning as dl_clf
from sklearn.metrics import accuracy_score, f1_score
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
import joblib
import os


def split_data(dataset, forecast_horizon):
    dataset = dataset.dropna()
    dataset_train, dataset_test = temporal_train_test_split(dataset, fh=forecast_horizon)
    dataset_train_features = dataset_train.loc[:,
                             ['Summary', 'Precip Type', 'Temperature (C)', 'Apparent Temperature (C)',
                              'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                              'Pressure (millibars)']]
    dataset_train_features.index = dataset_train_features.index.to_period(freq="h")
    dataset_train_features = dataset_train_features.dropna()
    dataset_train_target = dataset_train['Temperature (C)']
    dataset_train_target.index = dataset_train_target.index.to_period(freq="h")
    dataset_train_target = dataset_train_target.dropna()
    return dataset_train_features, dataset_train_target, dataset_test


def classification_model(dataset_train_features, dataset_test, scores, target_variable):
    # Drop target column from training & test sets
    X_train = dataset_train_features.drop(columns=[target_variable])
    Y_train = dataset_train_features[[target_variable]]

    X_test = dataset_test.drop(columns=[target_variable])
    Y_test = dataset_test[[target_variable]]

    # One hot encode categorical columns
    # Target label will be exluded from this below. (E.g. if target is 'Precip Type' this wont be encoded.
    categorical_cols = []
    if 'Summary' in X_train.columns:
        categorical_cols.append('Summary')
    if 'Precip Type' in X_train.columns:
        categorical_cols.append('Precip Type')

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Enforce boolean types as float to prevent exogeneous errors
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Force columns to numeric to avoid exogeneous X errors
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Label encode the target
    labelEncoder = LabelEncoder()
    Y_train = labelEncoder.fit_transform(Y_train.values.ravel())  # ravel to avoid data convergence warnings
    Y_test = labelEncoder.fit_transform(Y_test.values.ravel())

    # Translate data into type 'numpy'
    X_train_np = X_train.to_numpy()
    Y_train_np = Y_train  # already NumPy after label encoding
    X_test_np = X_test.to_numpy()
    Y_test_np = Y_test

    # Call sliding windows to parse data into timeseries groups
    window_size = 2
    X_train_window, Y_train_window = sliding_windows(X_train_np, Y_train_np, window_size)
    X_test_window, Y_test_window = sliding_windows(X_test_np,  Y_test_np,  window_size)

    # Convert the numpy dataframe into a nested panel dataframe for use with LSTMFCNClassifier
    X_train_lstm = from_3d_numpy_to_nested(X_train_window)
    X_test_lstm = from_3d_numpy_to_nested(X_test_window)

    # Build LSTM classifier
    lstm_clf = dl_clf.LSTMFCNClassifier(n_epochs=10)
    lstm_clf.fit(X_train_lstm, Y_train_window)

    # Prediction
    Y_predict = lstm_clf.predict(X_test_lstm)
    print(f"PREDICTION FOR VARIABLE {target_variable}: {Y_predict}")
    accuracy = accuracy_score(Y_test_window, Y_predict)
    f1 = f1_score(Y_test_window, Y_predict, average='weighted')
    print(f"[Classification - {target_variable}]\nAccuracy = {accuracy:.4f}\nF1 = {f1:.4f}")

    # Add scores to list
    scores = scores._append({
        'Target': target_variable,
        'Model': 'Classification-LSTMFCN',
        'Mean Absolute Error': None,
        'Root Mean Squared Error': None,
        'Mean Absolute Percentage Error': None,
        'Accuracy': accuracy,
        'F1-Score': f1
    }, ignore_index=True)

    # save model for use later
    joblib.dump(lstm_clf, 'LSTM_Classifier_Model')

    return scores


# Needed to parse data into 'windowed' groups for use with classifier LSTM
def sliding_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size + 1):
        X_slice = X[i: i + window_size]  # shape (window_size, D)
        # Label for the entire window (use the label of the last time step, or whatever is domain-appropriate)
        y_window = y[i + window_size - 1]

        X_windows.append(X_slice)
        y_windows.append(y_window)

    return np.array(X_windows), np.array(y_windows)


def regression_model(dataset, forecast_horizon, scores, target_variable):
    # convert categorical values into numerical values
    leSummary = LabelEncoder()
    leSummary.fit(dataset['Summary'])
    replaceSummary = leSummary.transform(dataset['Summary'])
    dataset['Summary'] = replaceSummary
    lePrecipType = LabelEncoder()
    lePrecipType.fit(dataset['Precip Type'])
    replacePrecipType = lePrecipType.transform(dataset['Precip Type'])
    dataset['Precip Type'] = replacePrecipType

    # Clone dataset before carrying out the following to

    # drop Unnecessary Column ( "Loud Column has 0 values")
    if 'Loud Cover' in dataset.columns:
        dataset.drop(columns=['Loud Cover'], inplace=True)

    # drop daily summary column as it has overlapping data
    if 'Daily Summary' in dataset.columns:
        dataset.drop(columns=['Daily Summary'], inplace=True)

    # normalise values using min max scaling in case it already hasn't been normalised
    dataset = (dataset - dataset.mean()) / dataset.std()

    # calling def_split to split the dataset up
    dataset_train_features, dataset_train_target, dataset_test = split_data(dataset, forecast_horizon)

    # build the LSTM model
    modelLSTM = NeuralForecastLSTM(max_steps=50)  # limit to 50 iterations for testing. Was originally default (1000)
    modelLSTM.fit(y=dataset_train_target,
                  X=dataset_train_features,
                  fh=forecast_horizon)

    # make predictions
    lstmPredictions = modelLSTM.predict(fh=forecast_horizon)
    print(f"PREDICTION FOR TARGET VARIABLE: {target_variable}: {lstmPredictions}")

    # calculate scores based off predictions
    lstmMAE = mean_absolute_error(dataset_test[target_variable], lstmPredictions)
    lstmRMSE = mean_squared_error(dataset_test[target_variable], lstmPredictions, square_root=True)
    lstmMAPE = mean_absolute_percentage_error(dataset_test[target_variable], lstmPredictions)

    print(f"[Regression - {target_variable}]\nMAE = {lstmMAE:.4f}\nRMSE = {lstmRMSE:.4f}\nMAPE = {lstmMAPE:.4f}")

    # save scores to panda dataframe
    scores = scores._append({'Target': target_variable,
                             'Model': 'Regression_LSTM',
                             'Mean Absolute Error': lstmMAE,
                             'Root Mean Squared Error': lstmRMSE,
                             'Mean Absolute Percentage Error': lstmMAPE,
                             'Accuracy': None,
                             'F1-Score': None},
                            ignore_index=True)

    # save model for use later
    joblib.dump(modelLSTM, 'LSTM_Model')

    ''' ** TEMP COMMENT TO SOLVE ISSUES **
    
    # build the TBATS model
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
                             'Mean Absolute Percentage Error': tbatsMAPE,
                             'Accuracy': None,
                             'F1-Score': None},
                            ignore_index=True)

    # save model for use later
    joblib.dump(modelTBATS, 'TBATS_Model')
    '''

    # return scores
    return scores


# Determines which model is run based on data at hand
# Determines which model is run based on data at hand
def model_selection(dataset, target_variable, dataset_train_features, dataset_train_target, dataset_test,
                    forecast_horizon, scores):
    """
    Chooses regression if we're predicting values like temperature, rainfall etc.,
    Classification if we're predicting Summary, precipitation type etc.
    """
    classification_variables = ['Summary', 'Precip Type']
    regression_variables = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
                            'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
    if target_variable in regression_variables:
        scores = regression_model(
            dataset=dataset,
            forecast_horizon=forecast_horizon,
            scores=scores,
            target_variable=target_variable
        )
    elif target_variable in classification_variables:
        scores = classification_model(
            dataset_train_features=dataset_train_features,
            dataset_test=dataset_test,
            scores=scores,
            target_variable=target_variable
        )
    else:
        raise ValueError(
            f"Unknown target: {target_variable}."
        )

    return scores


def plot_results(scores):
    metrics = [
        'Mean Absolute Error',
        'Root Mean Squared Error',
        'Mean Absolute Percentage Error',
        'Accuracy',
        'F1-Score'
    ]

    os.makedirs('model_graphs/model_results', exist_ok=True)
    # Group df by Target column
    grouped = scores.groupby('Target')

    for target_variable, grouped_df in grouped:
        models = grouped_df['Model'].unique().tolist()

        # Prepare box plot data
        box_data = []
        for model in models:
            row = grouped_df[grouped_df['Model'] == model].iloc[0]

            # Collect the metric values in order (fill missing with 0 or skip)
            row_values = []
            for metric in metrics:
                value = row[metric]
                if pd.isna(value):
                    value = 0
                row_values.append(float(value))
            box_data.append(row_values)
        box_data = np.array(box_data, dtype=float)

        # X-axis for metrics
        x = np.arange(len(metrics))
        bar_width = 0.8 / len(models)
        plt.figure(figsize=(10, 6))
        # Rows as groups of bars
        for i, model in enumerate(models):
            offset = i * bar_width
            plt.bar((x + offset).tolist(),
                    box_data[i],
                    width=bar_width,
                    label=model,
                    edgecolor='black')
        plt.ylabel('Score Value')
        plt.title(f"Scores for target variable: {target_variable}")
        plt.legend()
        plt.tight_layout()
        sanitised_filename = target_variable.replace(' ', '_').replace('/', '_')
        filename = f"{sanitised_filename}_scores.png"
        save_path = os.path.join('model_graphs', 'model_results', filename)
        plt.savefig(save_path)
        plt.show()


def main():
    # Load dataset & initialise forecast horizon
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Path of current file
    csv_path = os.path.join(current_dir, '..', '..', 'CSV_Files',
                            'weatherHistory_p.csv')  # This can be easily amended to consider all submitted data files
    dataset = pd.read_csv(csv_path)

    # Sets the period index straight away to avoid attribute errors for 'to_period'
    dataset['Date'] = pd.to_datetime(
        dataset['Date'].astype(str).str.replace(r'\+02:00', '', regex=True),
        errors='coerce'
    )
    dataset.drop_duplicates(subset=['Date'], inplace=True)
    if 'Date' in dataset.columns:
        dataset.set_index('Date', inplace=True)
    dataset = dataset.asfreq('h')

    forecast_horizon = ForecastingHorizon(np.arange(1, 25))

    # Define scoring metrics
    scores = pd.DataFrame(columns=[
        'Model',
        'Mean Absolute Error',
        'Root Mean Squared Error',
        'Mean Absolute Percentage Error',
        'Accuracy',
        'F1-Score'
    ])

    # Split data into training & test sets
    dataset_train_features, dataset_train_target, dataset_test = split_data(dataset, forecast_horizon)

    ''' ***TEMP COMMENTED OUT
    scores = regression_model(
        dataset=dataset,
        forecast_horizon=forecast_horizon,
        scores=scores
    )
    '''

    all_scores = pd.DataFrame(columns=[
        'Model',
        'Mean Absolute Error',
        'Root Mean Squared Error',
        'Mean Absolute Percentage Error',
        'Accuracy',
        'F1-Score'
    ])

    # ***TEST OF ADDING MODEL SELECTION FUNCTIONALITY IN INSTEAD OF JUST REGRESSION ***
    variables = ['Summary', 'Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
    for variable in variables:
        scores = model_selection(dataset, variable, dataset_train_features, dataset_train_target, dataset_test,
                                 forecast_horizon, scores)
        all_scores = all_scores._append(scores, ignore_index=True)

    # Plot results using matplotlib
    plot_results(all_scores)

    return all_scores


if __name__ == "__main__":
    main()
