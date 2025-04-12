import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Data
data = pd.read_csv("gesture_data.csv")
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode Gesture Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save gesture classes
np.save("classes.npy", label_encoder.classes_)
print("Saved gesture classes!")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile & Train Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Save Model
model.save("gesture_model.h5")
print("Model training complete!")
