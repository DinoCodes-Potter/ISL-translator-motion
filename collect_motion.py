import os
# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import mediapipe as mp
import numpy as np

# --- Configuration ---
DATA_PATH = os.path.join('ISL_Data') 
# Add your signs here. 'NOTHING' is critical to teach the AI baseline state.
actions = np.array(['HELP', 'THANK_YOU', 'NOTHING']) 
no_sequences = 30 # Number of video clips per sign
sequence_length = 30 # Frames per clip (approx 1 sec)

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Setup Folders ---
for action in actions:
    for sn in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sn)), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Starting Data Collection Script...")
print("Controls: 's' to start recording a sequence, 'q' to quit entirely.")

for action in actions:
    for sn in range(no_sequences):
        # 1. Waiting Loop (Setup your position)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Show instructions on screen
            cv2.putText(frame, f'NEXT: {action} (Video {sn})', (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Press "s" to START recording', (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Motion Collector', frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                break
            if key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # 2. Recording Loop (Burst capture 30 frames)
        print(f"Recording {action} sequence {sn}...")
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # MediaPipe requires RGB conversion
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            
            # Prepare data row (126 values: 2 hands * 21 landmarks * 3 coordinates)
            full_data = np.zeros(126)
            
            if res.multi_hand_landmarks:
                for i, hand_lms in enumerate(res.multi_hand_landmarks):
                    if i < 2: # Limit processing to 2 hands
                        # Extract landmark coordinates
                        for j, lm in enumerate(hand_lms.landmark):
                            idx = (i * 63) + (j * 3)
                            full_data[idx:idx+3] = [lm.x, lm.y, lm.z]
                        
                        # DRAW LANDMARKS (Critical for verifying data quality)
                        mp_drawing.draw_landmarks(
                            frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Save the landmark data as a .npy file
            npy_path = os.path.join(DATA_PATH, action, str(sn), str(frame_num))
            np.save(npy_path, full_data)

            # Visual feedback during the recording burst
            cv2.putText(frame, f'RECORDING: {action}', (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Frame: {frame_num}', (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.imshow('Motion Collector', frame)
            cv2.waitKey(10) # 10ms pause ensures frames are captured at a realistic pace

print("Data collection complete!")
cap.release()
cv2.destroyAllWindows()
