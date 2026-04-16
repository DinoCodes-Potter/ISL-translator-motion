import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

# --- 1. Load Data ---
DATA_PATH = os.path.join('ISL_Data')
actions = np.array(['HELP', 'THANK_YOU', 'NOTHING'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading data into memory...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences) # Shape: (90, 30, 126)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 2. Build the LSTM Model ---
model = Sequential()
# First LSTM layer needs input_shape (frames, points)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Output layer (3 neurons for 3 signs)
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 3. Train ---
print("Starting training... This will take a few minutes on CPU.")
model.fit(X_train, y_train, epochs=200) 

model.summary()

# --- 4. Save the Brain ---
model.save('isl_motion_model.h5')
print("Model saved as isl_motion_model.h5")
