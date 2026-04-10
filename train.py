import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
X = np.load('processed_data_euroc/X.npy', mmap_mode='r')
y = np.load('processed_data_euroc/y.npy', mmap_mode='r')

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(100, 9)),
    Dropout(0.3),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train (use subset to be fast)
model.fit(X[:20000], y[:20000], epochs=5, batch_size=32)

# Save model
import os
os.makedirs('models', exist_ok=True)
model.save('models/lstm_euroc_spec.h5')

print("Model saved!")
