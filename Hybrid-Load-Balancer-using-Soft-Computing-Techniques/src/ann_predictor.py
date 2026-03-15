import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import os

class LoadPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'models/predictor.keras' 

    def train_from_csv(self, file_path):
        if not os.path.exists('models'): os.makedirs('models')
        
        df = pd.read_csv(file_path)
        
        # --- FIX 1: Use the Google Data column name ---
        # The translator saves it as 'Requests_per_Minute'
        data = self.scaler.fit_transform(df[['Requests_per_Minute']])
        
        X, y = [], []
        for i in range(60, len(data)):
            X.append(data[i-60:i, 0])
            y.append(data[i, 0])
        
        self.model = Sequential([
            Input(shape=(60,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss=MeanSquaredError())
        self.model.fit(np.array(X), np.array(y), epochs=5, verbose=1) # verbose=1 to see progress
        self.model.save(self.model_path)
        print(f"Model trained on Google patterns and saved at {self.model_path}")

    def predict_next(self, history):
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                df = pd.read_csv(r'D:\LoadBalancer\Hybrid-Load-Balancer-using-Soft-Computing-Techniques\data\traffic_logs.csv')
                self.scaler.fit(df[['Requests_per_Minute']])
                
            else:
                return 50 # Default if no model exists
        
        # --- FIX 2: Wrap history in a DataFrame to fix 'Feature Names' Warning ---
        history_array = np.array(history).reshape(-1, 1)
        history_df = pd.DataFrame(history_array, columns=['Requests_per_Minute'])
        
        # Transform using the named DataFrame
        hist_scaled = self.scaler.transform(history_df)
        
        # Reshape for the ANN Input (1 sample, 60 timesteps)
        pred_input = hist_scaled.reshape(1, 60)
        pred = self.model.predict(pred_input, verbose=0)
        
        # Invert the scaling to get the real request count
        return self.scaler.inverse_transform(pred)[0][0]