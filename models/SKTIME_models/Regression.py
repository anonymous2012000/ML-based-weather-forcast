import os.path
import requests
from app import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
import joblib


def preprocess_data(file_path):
    """Load and preprocess the dataset."""
    dataset = pd.read_csv(file_path)
    dataset = dataset.rename(columns={'Date': 'Datetime'})
    dataset['Datetime'] = pd.to_datetime(dataset['Datetime'].astype(str).str.replace(r'\+02:00', '', regex=True),
                                         errors='coerce')

    # Drop unnecessary columns
    dataset.drop(columns=['Loud Cover', 'Daily Summary'], errors='ignore', inplace=True)

    # Encode categorical variables
    leSummary = LabelEncoder()
    dataset['Summary'] = leSummary.fit_transform(dataset['Summary'])

    lePrecipType = LabelEncoder()
    dataset['Precip Type'] = lePrecipType.fit_transform(dataset['Precip Type'])

    # Remove duplicates and set Datetime as index
    dataset = dataset.drop_duplicates(subset=['Datetime']).set_index('Datetime').asfreq('h')

    # Normalize data
    dataset = (dataset - dataset.mean()) / dataset.std()
    dataset = dataset.dropna()

    return dataset


def train_lstm_model(dataset):
    """Train an LSTM model on the given dataset and save it."""
    forecast_horizon = ForecastingHorizon(np.arange(1, 25))
    dataset_train, dataset_test = temporal_train_test_split(dataset, fh=forecast_horizon)

    # Select features and target
    features = ['Summary', 'Precip Type', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                'Pressure (millibars)']
    dataset_train_features = dataset_train[features]
    dataset_train_target = dataset_train['Temperature (C)']

    # Convert index to period
    dataset_train_features.index = dataset_train_features.index.to_period(freq="h")
    dataset_train_target.index = dataset_train_target.index.to_period(freq="h")

    # Train the model
    modelLSTM = NeuralForecastLSTM()
    modelLSTM.fit(y=dataset_train_target, X=dataset_train_features, fh=forecast_horizon)

    # Make predictions
    lstmPredictions = modelLSTM.predict(fh=forecast_horizon)

    # Evaluate model
    scores = pd.DataFrame([{
        'Model': 'LSTM',
        'Mean Absolute Error': mean_absolute_error(dataset_test['Temperature (C)'], lstmPredictions),
        'Root Mean Squared Error': mean_squared_error(dataset_test['Temperature (C)'], lstmPredictions,
                                                      square_root=True),
        'Mean Absolute Percentage Error': mean_absolute_percentage_error(dataset_test['Temperature (C)'],
                                                                         lstmPredictions)
    }])

    print(scores)

    # Save model
    model_path = joblib.dump(modelLSTM, 'LSTM_Model.pkl')[0]
    #joblib.dump(modelLSTM, 'LSTM_Model.pkl')
    print("Model saved as 'LSTM_Model.pkl'")
    #model_path = os.path.join(os.getcwd(), 'LSTM_Model.pkl')

    return model_path


def train(filename):
    filepath = os.path.join(app.config['TRAINING_DATA'], filename)
    dataset = preprocess_data(filepath)
    model_path = train_lstm_model(dataset)
    return model_path


def retrain(filename):
    dataset = preprocess_data(filename)
    model_path = train_lstm_model(dataset)
    return model_path

def store_model_ipfs(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"Error: File not found at {model_path}")
            return None

        url = "http://127.0.0.1:5002/api/v0/add"

        with open(model_path, "rb") as file:
            files = {"file": file}
            # Send the file to IPFS API
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                cid = result["Hash"]  # Get the CID
                cid_bytes = cid.encode("utf-8")
                print(f"File uploaded successfully, the CID: {cid}")
                print(f"Access the model on IPFS at: https://ipfs.io/ipfs/{cid}")
                return cid_bytes
            else:
                print(f"Error uploading file: {response.text}")
                return None

    except Exception as e:
        print(f"Error uploading model.pkl to IPFS: {e}")
        return None


if __name__ == '__main__':
    model_path = train('weatherHistory.csv')
    store_model_ipfs(model_path)
