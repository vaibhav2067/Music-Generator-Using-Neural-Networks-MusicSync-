from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

app = Flask(__name__)

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

# Serve index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to generate music
@app.route('/generate-music', methods=['POST'])
def generate_music_route():
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
        generated_sequence = generate_music(model, seed_sequence, temperature=temperature)

        output_file = 'generated_music.mid'
        sequence_to_midi(generated_sequence, output_file)

        return jsonify(success=True, fileUrl=f"/download/{output_file}")

    except Exception as e:
        return jsonify(success=False, error=str(e))

# Route to download generated music
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)
 
if __name__ == '__main__':
    app.run(debug=True)
