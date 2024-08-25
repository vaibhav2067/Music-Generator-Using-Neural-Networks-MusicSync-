import os
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Parameters
vocab_size = 128
sequence_length = 50
embedding_dim = 64
epochs = 10
batch_size = 64
generated_length = 100
folder_path = "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/Classic"  # Replace with the path to your folder

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
    
    # Check if the number of notes is less than the sequence length
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
        if sequence is not None and len(sequence) > 0:  # Ensure the sequence is valid
            X.append(sequence)
    
    if len(X) == 0:
        raise ValueError("No valid MIDI files found or all failed to process.")
    
    # Ensure that all sequences are concatenated properly
    X = np.concatenate([x for x in X if x.ndim == 2], axis=0)
    return X

try:
    X_train = prepare_data_from_folder(folder_path, sequence_length)
    y_train = X_train[:, -1]
    X_train = X_train[:, :-1]
except ValueError as e:
    print(e)
    exit()

# Build and train the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Generate music
def generate_music(model, seed_sequence, length=generated_length):
    generated = seed_sequence
    for _ in range(length):
        prediction = model.predict(np.expand_dims(generated, axis=0))
        next_note = np.argmax(prediction[0], axis=-1)
        generated = np.append(generated[1:], next_note)
    return generated

seed_sequence = X_train[np.random.randint(0, len(X_train))]
generated_sequence = generate_music(model, seed_sequence)

# Convert generated sequence to MIDI
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

sequence_to_midi(generated_sequence, 'generated_music.mid')

print("Music generation complete. Check 'generated_music.mid' for the result.")
