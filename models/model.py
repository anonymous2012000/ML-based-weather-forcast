import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# **1. Load Data**
df = pd.read_csv('/content/sample_data/weatherHistory.csv', parse_dates=['Date'])

# **2. Convert Date Format**
df['Date'] = pd.to_datetime(df['Date'].astype(str).str.replace(r'\+02:00', '', regex=True), errors='coerce')

# **3. Drop Unnecessary Column ( "Loud Column has 0 values")**
if 'Loud Column' in df.columns:
    df.drop(columns=['Loud Column'], inplace=True)

# **4. Select Relevant Columns**
features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 
            'Visibility (km)', 'Pressure (millibars)']
target = df[['Temperature (C)']]

df = df[['Date'] + features].dropna()

# **5. Normalize Features and Target Separately**
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df_features = df[features]
df_target = df[target]

df_features_scaled = scaler_X.fit_transform(df_features)
df_target_scaled = scaler_y.fit_transform(target)

# **6. Create Sequences for LSTM**
def create_sequences(features, target, n_steps):
    X, y = [], []
    for i in range(len(features) - n_steps):
        X.append(features[i:i + n_steps])
        y.append(target[i + n_steps + 1])  # Predict next temperature
    return np.array(X), np.array(y)

n_steps = 24  # 24-hour window

X, y = create_sequences(df_features_scaled, df_target_scaled, n_steps)

# **7. Split Data**
test_size = 24  # Last 24 hours for testing
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

# **8. Build LSTM Model**
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, X.shape[2])),
    Dropout(0.3),
    LSTM(50, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Predicting temperature only
])

model.compile(optimizer='adam', loss='mse')

# **9. Callbacks for Early Stopping and Model Checkpointing**
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# **10. Train Model**
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), 
                    callbacks=[early_stop, checkpoint], batch_size=32)

# **11. Plot Training History**
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# **12. Evaluate on Test Set**
predictions = model.predict(X_test)

# **13. Inverse Transform Predictions and Actual Values**
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
predictions_actual = scaler_y.inverse_transform(predictions).flatten()

# **14. Plot Predictions vs Actual Temperature**
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Temperature', linestyle='dashed')
plt.plot(predictions_actual, label='Predicted Temperature')
plt.title('24-Hour Temperature Prediction')
plt.xlabel('Hours')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# **15. Save Model and Scalers**
model.save('temperature_predictor.h5')
joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')
