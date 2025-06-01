import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM

def regression_predict(user_csv):
    # Load the trained LSTM model
    modelLSTM = joblib.load('LSTM_Model.pkl')


    # Read the user-supplied CSV (expects one row)
    user_data = pd.read_csv(user_csv)


    # Rename columns and process datetime
    user_data = user_data.rename(columns={'Date': 'Datetime'})
    user_data['Datetime'] = pd.to_datetime(user_data['Datetime'].astype(str).str.replace(r'\+02:00', '', regex=True), errors='coerce')

    # Drop unnecessary columns to match original datset
    if 'Loud Cover' in user_data.columns:
        user_data.drop(columns=['Loud Cover'], inplace=True)
    if 'Daily Summary' in user_data.columns:
        user_data.drop(columns=['Daily Summary'], inplace=True)


    # Convert categorical values to numerical values same as above
    leSummary = LabelEncoder()
    leSummary.fit(user_data['Summary'])
    user_data['Summary'] = leSummary.fit_transform(user_data['Summary'])
    lePrecipType = LabelEncoder()
    lePrecipType.fit(user_data['Precip Type'])
    user_data['Precip Type'] = lePrecipType.fit_transform(user_data['Precip Type'])

    # Drop duplicates and set datetime index
    user_data = user_data.drop_duplicates(subset=['Datetime'])
    user_data = user_data.set_index('Datetime').asfreq('h')

    # using this as a fix as it was difficult to get the std for a single row
    train_stats = {
        "Summary": {"mean": 1.5, "std": 0.5},
        "Precip Type": {"mean": 0.3, "std": 0.4},
        "Humidity": {"mean": 0.7, "std": 0.1},
        "Wind Speed (km/h)": {"mean": 10.0, "std": 3.0},
        "Wind Bearing (degrees)": {"mean": 180.0, "std": 90.0},
        "Visibility (km)": {"mean": 10.0, "std": 5.0},
        "Pressure (millibars)": {"mean": 1015.0, "std": 10.0},
        "Temperature (C)": {"mean": 15.0, "std": 5.0},  # Needed for reverse normalization
    }

    # Apply normalization using training stats
    for col in user_data.columns:
        if col in train_stats:
            mean_val = train_stats[col]["mean"]
            std_val = train_stats[col]["std"]

            if std_val == 0:  # Prevent division by zero
                user_data[col] = 0
            else:
                user_data[col] = (user_data[col] - mean_val) / std_val

    # Normalize using mean and standard deviation from input itself, this has led to unending errors because of std and one row ish
   # user_data = (user_data - user_data.mean()) / user_data.std()


    # Select features used during training
    user_features = user_data[['Summary', 'Precip Type', 'Humidity', 'Wind Speed (km/h)',
                               'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']]
    

    # Set forecasting horizon
    fh = ForecastingHorizon(np.arange(1,25))

    

    # Predict temperature
    prediction = modelLSTM.predict(fh=fh, X=user_features)

    # Reverse the normalization to get original scale
    #predicted_temperature = (prediction)
    predicted_temperature = (prediction * train_stats["Temperature (C)"]["std"]) + train_stats["Temperature (C)"][
        "mean"]
    #returns 24 values due to FH=25, getting only the first prediction for now, more elegant solution might be to get average of 24 values
    first_prediction = predicted_temperature.iloc[0]

    #return predicted_temperature
    return first_prediction.round()



def main():
    """ Main function to run the prediction """
    user_csv = "Predict_CSV.csv" # Path to the user's input CSV file,use same file as datset and edit a row


    # Run the prediction
    predicted_temp = regression_predict(user_csv)

    # Output the result
    print("Final Predicted Temperature (°C):")
    print(predicted_temp)



if __name__ == '__main__':
    main()
