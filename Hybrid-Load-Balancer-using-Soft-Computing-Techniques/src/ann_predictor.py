import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
import os

WINDOW_SIZE = 60

class LoadPredictor:
    def __init__(self):
        self.model = None
        self.model_path = 'models/predictor_v2.keras'

    def train_from_df(self, df):
        if not os.path.exists('models'): os.makedirs('models')
        data = df['request_count'].values 
        X, y = [], []
        for i in range(WINDOW_SIZE, len(data)):
            X.append(data[i - WINDOW_SIZE:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        self.model = Sequential([
            Input(shape=(WINDOW_SIZE,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1,  activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss=MeanSquaredError())
        self.model.fit(X, y, epochs=15, batch_size=32, verbose=0)
        self.model.save(self.model_path)

    def predict_next(self, history):
        if self.model is None:
            if os.path.exists(self.model_path): self.model = load_model(self.model_path)
            else: return 0.5
        history = list(history)
        if len(history) < WINDOW_SIZE:
            history = [np.mean(history) if history else 0.5] * (WINDOW_SIZE - len(history)) + history
        hist_array = np.array(history[-WINDOW_SIZE:], dtype=np.float32).reshape(1, WINDOW_SIZE)
        pred = self.model.predict(hist_array, verbose=0)
        return float(pred[0][0])

    def compute_trend_score(self, history):
        """
        NASA-Style Smoothing Logic:
        We use a rolling mean of the history to detect the 'Vibe' 
        rather than the 'Jitter'.
        """
        raw_pred = self.predict_next(history)
        
        # Smooth the recent history to find the underlying wave
        recent_avg = np.mean(history[-10:]) if len(history) >= 10 else raw_pred
        long_term_avg = np.mean(history) if len(history) > 0 else raw_pred
        
        # --- THE "NASA WAVE" MATH ---
        # Base visibility: line starts at 10%
        base_visibility = 10.0
        
        # Volume factor: scale the raw prediction up to 50
        volume_factor = raw_pred * 50.0
        
        # Volatility factor: This creates the "spikes" in the red line 
        # by comparing recent movement to the long-term baseline.
        volatility = abs(recent_avg - long_term_avg) * 400.0
        
        # Combine them into a dynamic score
        trend_score = base_visibility + volume_factor + volatility
        
        return float(np.clip(trend_score, 0, 100))

    def load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
