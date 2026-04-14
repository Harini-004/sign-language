import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

"""
English Sign Language Real-time Testing

This script uses the trained model to perform real-time sign language recognition
from webcam input. It implements several improvements for better accuracy:

1. Proper sequence collection for temporal information
2. Prediction smoothing to reduce jitter
3. Confidence threshold for unknown gestures
4. Improved visualization
"""

# Path settings
MODEL_PATH = os.path.join('models', 'english_sign_language.h5')

# Load actions from the data directory or use default
DATA_PATH = os.path.join('data')
actions = np.array([
    "How are you", "Have a nice day", "Good morning","Be positive","All the best"
])

# Define colors for visualization
colors = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255)
]

# Ensure we have enough colors
while len(colors) < len(actions):
    colors.extend(colors[:len(actions)-len(colors)])

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

# Function to visualize probabilities of actions
def prob_viz(actions, res, input_frame, colors, threshold=0.7):
    output_frame = input_frame.copy()
    
    # Draw bounding boxes for each action with improved visibility
    for num, prob in enumerate(res):
        if num < len(actions) and num < len(colors):
            # Make bar width proportional to probability
            bar_width = int(prob * 300)  # Increased width for better visibility
            cv2.rectangle(output_frame, (0, 60 + num * 40), (bar_width, 90 + num * 40), colors[num], -1)
            
            # Add text label with probability percentage
            label = f"{actions[num]}: {prob:.2f}"
            cv2.putText(output_frame, label, (bar_width + 10, 80 + num * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Get the index of the action with the highest probability
    max_prob_index = np.argmax(res)
    max_prob = res[max_prob_index]
    
    # Get the predicted action and its corresponding probability
    predicted_action = actions[max_prob_index] if max_prob >= threshold else 'Unknown'

    # Display the prediction
    text = f'Prediction: {predicted_action} ({max_prob:.2f})'
    cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return output_frame

# Function to apply temporal smoothing to predictions
def smooth_predictions(predictions, window_size=5):
    if len(predictions) < window_size:
        return np.zeros(len(actions)) if len(predictions) == 0 else predictions[-1]
    
    # Average the last window_size predictions
    recent_preds = np.array(predictions[-window_size:])
    return np.mean(recent_preds, axis=0)

# Main function to run real-time prediction
def run_realtime_prediction():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run 2_model_training.py first to train the model.")
        return
    
    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Print model input shape to verify
    print(f"Model input shape: {model.input_shape}")
    
    # Initialize variables
    sequence = []
    predictions = []
    threshold = 0.7  # Confidence threshold for detection
    
    # Set up webcam feed
    cap = cv2.VideoCapture(0)
    
    # Create a named window
    cv2.namedWindow('English Sign Language Recognition', cv2.WINDOW_NORMAL)
    
    # Main loop
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Make detection
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Append keypoints to sequence
            sequence.append(keypoints)
            
            # Keep only the last frames (matching model's expected sequence length)
            sequence_length = model.input_shape[1]  # Get expected sequence length from model
            if len(sequence) > sequence_length:
                sequence = sequence[-sequence_length:]
            
            # Only predict when we have enough frames
            if len(sequence) == sequence_length:
                # Prepare input for model
                input_data = np.expand_dims(np.array(sequence), axis=0)
                
                # Make prediction
                res = model.predict(input_data, verbose=0)[0]
                
                # Add prediction to history for smoothing
                predictions.append(res)
                
                # Apply smoothing to reduce jitter
                smoothed_res = smooth_predictions(predictions)
                
                # Visualize prediction
                image = prob_viz(actions, smoothed_res, image, colors, threshold)
            
            # Show to screen
            cv2.imshow('English Sign Language Recognition', image)
            
            # Add help text
            help_text = "Press 'q' to quit"
            cv2.putText(image, help_text, (10, image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("English Sign Language Real-time Testing")
    print(f"Actions: {actions}")
    run_realtime_prediction()