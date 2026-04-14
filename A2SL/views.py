from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login,logout
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
import cv2
import numpy as np
import mediapipe as mp
from django.http import StreamingHttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import os
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('punkt')


# Define Tamil letters for recognition
actions = np.array(['hello how are you', 'good morning', '3', '4', '5'])

# Load the trained model (Ensure the correct path)
MODEL_PATH = r"C:\Users\harin\OneDrive\Desktop\deaf_and_dum\handdet\best_modelacc.keras"

def home_view(request):
	return render(request,'home.html')

@login_required(login_url="login")
def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		#tokenizing the sentence
		text.lower()
		#tokenizing the sentence
		words = word_tokenize(text)

		tagged = nltk.pos_tag(words)
		tense = {}
		tense["future"] = len([word for word in tagged if word[1] == "MD"])
		tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
		tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
		tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])



		#stopwords that will be removed
		stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])



		#removing stopwords and applying lemmatizing nlp process to words
		lr = WordNetLemmatizer()
		filtered_text = []
		for w,p in zip(words,tagged):
			if w not in stop_words:
				if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
					filtered_text.append(lr.lemmatize(w,pos='v'))
				elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
					filtered_text.append(lr.lemmatize(w,pos='a'))

				else:
					filtered_text.append(lr.lemmatize(w))


		#adding the specific word to specify tense
		words = filtered_text
		temp=[]
		for w in words:
			if w=='I':
				temp.append('Me')
			else:
				temp.append(w)
		words = temp
		probable_tense = max(tense,key=tense.get)

		if probable_tense == "past" and tense["past"]>=1:
			temp = ["Before"]
			temp = temp + words
			words = temp
		elif probable_tense == "future" and tense["future"]>=1:
			if "Will" not in words:
					temp = ["Will"]
					temp = temp + words
					words = temp
			else:
				pass
		elif probable_tense == "present":
			if tense["present_continuous"]>=1:
				temp = ["Now"]
				temp = temp + words
				words = temp


		filtered_text = []
		for w in words:
			path = w + ".mp4"
			f = finders.find(path)
			#splitting the word if its animation is not present in database
			if not f:
				for c in w:
					filtered_text.append(c)
			#otherwise animation of word
			else:
				filtered_text.append(w)
		words = filtered_text;


		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')




def signup_view(request):
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request,user)
			# log the user in
			return redirect('animation')
	else:
		form = UserCreationForm()
	return render(request,'signup.html',{'form':form})



def login_view(request):
	if request.method == 'POST':
		form = AuthenticationForm(data=request.POST)
		if form.is_valid():
			#log in user
			user = form.get_user()
			login(request,user)
			if 'next' in request.POST:
				return redirect(request.POST.get('next'))
			else:
				return redirect('animation')
	else:
		form = AuthenticationForm()
	return render(request,'login.html',{'form':form})


def logout_view(request):
	logout(request)
	return redirect("home")

# Path settings
model = load_model(r"C:\Users\Lenovo\Downloads\deaf_and_dum (2)\deaf_and_dum\models\english_sign_language.h5")
# Load actions from the data directory or use default
DATA_PATH = os.path.join('data')
actions = np.array([
    "How are you", "Have a nice day", "Good morning","Be positive","All the best"
])
# Define colors for visualization
colors = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),
	(212, 255, 0),      # Green
    (0, 212, 255),
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
    # Print model input shape to verify
print(f"Model input shape: {model.input_shape}")

# Initialize variables
  
def generate_frames():   
    sequence = []
    predictions = []
    threshold = 0.7
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

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def sign_language_view(request):
    return render(request, 'sign_language.html')