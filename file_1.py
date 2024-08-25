import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.utils import to_categorical

def preprocess_audio(file_path, n_mfcc=20, max_pad_len=130):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # Pad the sequences to have the same length
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_dir, genres, max_pad_len=130):
    X = []
    y = []
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        for file in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, file)
            mfcc = preprocess_audio(file_path, max_pad_len=max_pad_len)
            if mfcc is not None:
                X.append(mfcc)
                y.append(genres.index(genre))
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(genres))
    return X, y

# Example usage
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
data_dir = 'path_to_gtzan_data'
X, y = load_dataset(data_dir, genres)

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (X.shape[1], X.shape[2])
model = build_lstm_model(input_shape, len(genres))
model.summary()

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

def predict_genre(model, file_path, genres, max_pad_len=130):
    mfcc = preprocess_audio(file_path, max_pad_len=max_pad_len)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

# Example usage
test_file = 'path_to_test_audio_file.wav'
predicted_genre = predict_genre(model, test_file, genres)
print(f"The predicted genre is: {predicted_genre}")
