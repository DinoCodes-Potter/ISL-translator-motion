import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 1. Setup
actions = np.array(['HELP', 'THANK_YOU', 'NOTHING'])
model = load_model('isl_motion_model.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Variables for the "Rolling Window"
sequence = []
sentence = []
threshold = 0.8 # Only show result if model is > 80% sure

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    
    # Extract landmarks (126 values)
    full_data = np.zeros(126)
    if res.multi_hand_landmarks:
        for i, hand_lms in enumerate(res.multi_hand_landmarks):
            if i < 2:
                for j, lm in enumerate(hand_lms.landmark):
                    idx = (i * 63) + (j * 3)
                    full_data[idx:idx+3] = [lm.x, lm.y, lm.z]
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # RECOGNITION LOGIC -----------------------------------------
    sequence.append(full_data)
    sequence = sequence[-30:] # Keep only the last 30 frames

    if len(sequence) == 30:
        # Predict! (input shape needs to be 1, 30, 126)
        res_prob = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        # Get the sign with the highest probability
        if res_prob[np.argmax(res_prob)] > threshold:
            current_action = actions[np.argmax(res_prob)]
            
            # Display logic
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, f'GESTURE: {current_action} ({res_prob[np.argmax(res_prob)]*100:.1f}%)', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('ISL Real-Time Motion Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
