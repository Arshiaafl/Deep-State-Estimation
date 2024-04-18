import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

# Load dataset from CSV
df = pd.read_csv(
    r"C:\Users\ARASOFT\OneDrive\Desktop\PH.D\Deep Est\motor_speed_data.csv")

# Extract motor speed measurements
measurements = df['Motor_current'].values.reshape(-1, 1)

# Scale the data
scaler = StandardScaler()
scaled_measurements = scaler.fit_transform(measurements)

# Prepare data for LSTM (convert to supervised learning)
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, :])
    return np.array(X), np.array(y)

time_steps = 5  # Number of time steps (sequence length)
X_lstm, y_lstm = create_dataset(scaled_measurements, time_steps)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# LSTM model
#model = Sequential()
#model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dense(units=1))
#model.compile(optimizer='adam', loss='mean_squared_error')

#or, Load the model
from keras.models import load_model

# Load the saved LSTM model
model = tf.keras.models.load_model('lstm_motor_speed_model.h5')
#loaded_model.compile(optimizer='adam', loss=mean_squared_error)
# Train the model
#history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict using LSTM model
y_pred_lstm = model.predict(X_test)

# Inverse transform predictions
y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
y_test_inv = scaler.inverse_transform(y_test)

#model.save("lstm_motor_speed_model.h5")

# Kalman filter implementation for motor speed estimation
class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x_est = None  # State estimate
        self.P_est = None  # Error covariance

    def predict(self):
        # Prediction step
        self.x_est = np.dot(self.F, self.x_est)
        self.P_est = np.dot(np.dot(self.F, self.P_est), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x_est)
        S = np.dot(np.dot(self.H, self.P_est), self.H.T) + self.R
        K = np.dot(np.dot(self.P_est, self.H.T), np.linalg.inv(S))
        self.x_est = self.x_est + np.dot(K, y)
        self.P_est = self.P_est - np.dot(np.dot(K, self.H), self.P_est)

# Initialize Kalman filter parameters
dt = 1                      # Time step
F = np.array([[1, dt],     # State transition matrix
              [0, 1]])
H = np.array([[1, 0]])     # Measurement matrix
Q = np.eye(2) * 0.001      # Process noise covariance
R = np.eye(1) * 0.01       # Measurement noise covariance

# Initialize Kalman filter
kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
kf.x_est = np.array([[0], [0]])   # Initial state estimate
kf.P_est = np.eye(2)              # Initial error covariance

# Perform Kalman filter state estimation
estimated_speed_kf = []
for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    estimated_speed_kf.append(kf.x_est[0, 0])

# Plot results
plt.figure(figsize=(10, 5))
last_20_samples = len(y_test_inv) - 20  # Get the index of the last 20 samples

plt.plot(y_test_inv[last_20_samples:], label='True Motor Current', color='blue')
plt.plot(y_pred_lstm_inv[last_20_samples:], label='Predicted Motor Current (LSTM)', linestyle='--', color='green')
plt.plot(estimated_speed_kf[-20:], label='Estimated Motor Current (Kalman Filter)', linestyle='-.', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Motor Current (A)')
plt.title('Motor Current Estimation: LSTM vs Kalman Filter (Last 20 Samples)')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error

# Calculate RMSE for LSTM model
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))

# Calculate RMSE for Kalman filter
# Truncate estimated_speed_kf to have the same length as y_test_inv
estimated_speed_kf_truncated = estimated_speed_kf[-len(y_test_inv):]

# Calculate RMSE for Kalman filter
rmse_kalman = np.sqrt(mean_squared_error(y_test_inv, estimated_speed_kf_truncated))

print("RMSE for LSTM:", rmse_lstm)
print("RMSE for Kalman Filter:", rmse_kalman)
