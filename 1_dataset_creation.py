import cv2
import numpy as np
import mediapipe as mp
import os
import time

"""
English Sign Language Dataset Creation
This script captures hand gesture data for English sign language recognition.
It uses MediaPipe to extract hand landmarks and saves the data for model training.
"""

# Create necessary directories
DATA_PATH = os.path.join('data')
os.makedirs(DATA_PATH, exist_ok=True)

# Define actions/signs to capture
actions = np.array([
    "1"
])

# Create directories for each action
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    
# Number of sequences and frames per sequence
no_sequences = 30
sequence_length = 30

# Set up MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to perform detection using MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks on image
def draw_styled_landmarks(image, results):
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )

# Function to extract keypoints from MediaPipe results
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

# Main function to capture dataset
def capture_dataset():
    cap = cv2.VideoCapture(0)
    # Set up MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through actions
        for action in actions:
            # Loop through sequences
            for sequence in range(no_sequences):
                # Create directory for this sequence
                sequence_path = os.path.join(DATA_PATH, action, str(sequence))
                os.makedirs(sequence_path, exist_ok=True)
                # Display instructions
                print(f'Collecting data for {action}, sequence {sequence}')
                print('Prepare for recording in 3 seconds...')
                # Countdown
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Flip the frame horizontally for a more intuitive mirror view
                    frame = cv2.flip(frame, 1)
                    
                    # Display countdown
                    cv2.putText(frame, str(countdown), (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 4, cv2.LINE_AA)
                    
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(1000)
                
                # Start collecting frames for the sequence
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Flip the frame horizontally
                    frame = cv2.flip(frame, 1)
                    
                    # Make detection
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Display collection progress
                    cv2.putText(image, f'Collecting frames for {action} - Sequence {sequence} - Frame {frame_num}', 
                               (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_path, str(frame_num) + '.npy')
                    np.save(npy_path, keypoints)
                    
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
                # Short pause between sequences
                time.sleep(1)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Dataset collection complete!")

if __name__ == "__main__":
    print("English Sign Language Dataset Creation Tool")
    print("This will capture", len(actions), "signs with", no_sequences, "sequences of", sequence_length, "frames each.")
    print("Signs to capture:", ", ".join(actions))
    input("Press Enter to begin data collection...")
    capture_dataset()