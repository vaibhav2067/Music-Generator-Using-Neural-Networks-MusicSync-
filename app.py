from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit
import time
import os
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from google.oauth2 import id_token
from google.auth.transport import requests

app = Flask(__name__)
socketio = SocketIO(app)


# Parameters
vocab_size = 128
sequence_length = 50
embedding_dim = 64
epochs = 1
batch_size = 64
generated_length = 100

genre_folders = {
    "Classic": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Classic",
    "Pop": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Pop"
}

# Preprocess MIDI files
def midi_to_sequence(midi_file, sequence_length):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None
    
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)
    
    notes = np.array(notes)
    
    if len(notes) < sequence_length:
        print(f"Skipping {midi_file}: Sequence length {len(notes)} is shorter than required {sequence_length}")
        return None
    
    notes = notes % vocab_size
    
    sequences = []
    for i in range(len(notes) - sequence_length):
        sequences.append(notes[i:i + sequence_length])
    
    return np.array(sequences)

def prepare_data_from_folder(folder_path, sequence_length):
    midi_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mid')]
    X = []
    for midi_file in midi_files:
        sequence = midi_to_sequence(midi_file, sequence_length)
        if sequence is not None and len(sequence) > 0:
            X.append(sequence)
    
    if len(X) == 0:
        raise ValueError("No valid MIDI files found or all failed to process.")
    
    X = np.concatenate([x for x in X if x.ndim == 2], axis=0)
    return X

# Model creation
def create_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=sequence_length-1),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Sample with temperature
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Generate music
def generate_music(model, seed_sequence, length=generated_length, temperature=1.0):
    generated = seed_sequence
    for _ in range(length):
        prediction = model.predict(np.expand_dims(generated, axis=0))
        next_note = sample_with_temperature(prediction[0], temperature)
        generated = np.append(generated[1:], next_note)
    return generated

# Convert sequence to MIDI
def sequence_to_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    start_time = 0

    for note_value in sequence:
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(note_value),
            start=start_time,
            end=start_time + 0.5
        )
        instrument.notes.append(note)
        start_time += 0.5

    midi.instruments.append(instrument)
    midi.write(output_file)

@app.route('/google-signin', methods=['POST'])
def google_signin():
    data = request.json
    token = data.get('credential')
    try:
        # Specify the CLIENT_ID of the app that accesses the backend
        CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)
        
        # ID token is valid
        user_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']
        # Handle the authenticated user (e.g., save to the database)
        return jsonify(success=True, user={"id": user_id, "email": email, "name": name})
    except ValueError:
        # Invalid token
        return jsonify(success=False), 400

# Serve start
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get-started')
def get_started():
    return render_template('index.html')

# Mock for canceling music generation
generation_cancelled = False

# Route to generate music
@app.route('/generate-music', methods=['POST'])
def generate_music_route():
    global generation_cancelled
    try:
        genre = request.form['genre']
        temperature = float(request.form['temperature'])
        folder_path = genre_folders.get(genre)

        if not folder_path:
            return jsonify(success=False, error="Invalid genre selected.")

        X_train = prepare_data_from_folder(folder_path, sequence_length)
        y_train = X_train[:, -1]  # Target is the last note
        X_train = X_train[:, :-1]  # Input is all but the last note

        # Create and train the model
        model = create_model()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Generate music
        seed_sequence = X_train[np.random.randint(0, len(X_train))]
        total_steps = 10
        for step in range(total_steps):
            if generation_cancelled:
                break  # Stop generating if canceled
            
            # Simulate music generation step by step
            time.sleep(0.1)  # Simulate processing time for each step
            progress = (step + 1) * 10  # Update progress (10%, 20%, ..., 100%)
            
            # Emit progress via SocketIO
            socketio.emit('progress_update', {
                'progress': progress,
                'message': f"Generating... {progress}%",
                'fileUrl': '/download/generated_music.mid'
            })

        if not generation_cancelled:
            generated_sequence = generate_music(model, seed_sequence, temperature)
            output_file = 'generated_music.mid'
            sequence_to_midi(generated_sequence, output_file)

            return jsonify(success=True, fileUrl=f"/download/{output_file}")

        else:
            return jsonify(success=False, error="Generation was canceled")

    except Exception as e:
        return jsonify(success=False, error=str(e))

# Route to download generated music
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

# Route to cancel the generation process
@app.route('/cancel-generation', methods=['POST'])
def cancel_generation():
    global generation_cancelled
    generation_cancelled = True
    return jsonify(success=True)
 
if __name__ == '__main__':
    app.run(debug=True)
