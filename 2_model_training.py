import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

"""
English Sign Language Model Training

This script loads the dataset created by 1_dataset_creation.py,
builds an LSTM-based model, and trains it for sign language recognition.
"""

# Path settings
DATA_PATH = os.path.join('data')
model_path = os.path.join('models', 'english_sign_language.h5')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Load actions from the dataset directory
actions = np.array(os.listdir(DATA_PATH)) if os.path.exists(DATA_PATH) else np.array([
    "1","2","3","4","5"
])

# Dataset parameters
no_sequences = 30
sequence_length = 30
feature_length = 126  # 21 landmarks * 3 coordinates * 2 hands

def load_dataset():
    """
    Load the dataset from the data directory
    """
    sequences, labels = [], []
    
    # Loop through actions
    for action_idx, action in enumerate(actions):
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            print(f"Warning: Directory {action_dir} not found. Skipping.")
            continue
            
        # Loop through sequences
        for sequence in range(no_sequences):
            sequence_dir = os.path.join(action_dir, str(sequence))
            if not os.path.exists(sequence_dir):
                print(f"Warning: Sequence directory {sequence_dir} not found. Skipping.")
                continue
                
            # Load all frames in this sequence
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    frame = np.load(frame_path)
                    window.append(frame)
                else:
                    print(f"Warning: Frame {frame_path} not found. Using zeros.")
                    window.append(np.zeros(feature_length))
            
            # Add sequence and label
            sequences.append(window)
            labels.append(action_idx)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Dataset loaded: {X.shape} sequences, {y.shape} labels")
    return X, y

def build_model(input_shape, num_classes):
    """
    Build an LSTM model for sequence classification
    """
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    # Save figure
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', 'training_history.png'))
    plt.show()

def train_model():
    """
    Load dataset, build and train the model
    """
    # Load dataset
    X, y = load_dataset()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, feature_length)
    num_classes = len(actions)
    model = build_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    log_dir = os.path.join('logs')
    os.makedirs(log_dir, exist_ok=True)
    
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping]
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history

if __name__ == "__main__":
    print("English Sign Language Model Training")
    print(f"Actions: {actions}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory {DATA_PATH} not found.")
        print("Please run 1_dataset_creation.py first to create the dataset.")
    else:
        train_model()