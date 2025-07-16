import os
import numpy as np
import pretty_midi
from tensorflow.keras.models import Sequential, load_model #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Embedding #type: ignore
from tensorflow.keras.models import save_model #type: ignore

# -----------------------------------------------
# Constants
# -----------------------------------------------
vocab_size = 128
sequence_length = 50
embedding_dim = 64
epochs = 10
batch_size = 64
MODEL_DIR = 'saved_models'
SEED_DATA_DIR = 'saved_seed_data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SEED_DATA_DIR, exist_ok=True)

# -----------------------------------------------
# Genre Folder Paths
# -----------------------------------------------
genre_folders = {
    "Classic": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Classic",
    "Pop": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Pop",
    "Rock": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Rock",
    "Rap": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Rap",
    "Dance": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Dance"
}

# -----------------------------------------------
# Data & Model Utilities
# -----------------------------------------------
def midi_to_sequence(midi_file, sequence_length):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None
    notes = [note.pitch for instrument in midi.instruments for note in instrument.notes]
    notes = np.array(notes) % vocab_size
    if len(notes) < sequence_length:
        return None
    sequences = [notes[i:i + sequence_length] for i in range(len(notes) - sequence_length)]
    return np.array(sequences)

def prepare_data_from_folder(folder_path, sequence_length):
    midi_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mid')]
    X = [midi_to_sequence(f, sequence_length) for f in midi_files if midi_to_sequence(f, sequence_length) is not None]
    if not X:
        raise ValueError(f"No valid MIDI sequences found in {folder_path}")
    return np.concatenate(X, axis=0)

def create_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=sequence_length - 1),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def train_and_save_model_for_genre(genre, folder_path):
    print(f"\n--- Training model for genre: {genre} ---")
    X = prepare_data_from_folder(folder_path, sequence_length)
    y = X[:, -1]
    X = X[:, :-1]

    model = create_model()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    save_path = os.path.join(MODEL_DIR, f"{genre.lower()}_model.h5")
    model.save(save_path)
    print(f"✅ Saved model to {save_path}")

    # Generate and save seed data
    generate_and_save_seed_data(X, genre)

def generate_and_save_seed_data(X, genre):
    seed_path = os.path.join(SEED_DATA_DIR, f"{genre.lower()}_seed.npy")
    np.save(seed_path, X)
    print(f"✅ Saved seed data to {seed_path}")

# -----------------------------------------------
# Main Execution
# -----------------------------------------------
if __name__ == '__main__':
    for genre, path in genre_folders.items():
        if os.path.exists(path):
            model_path = os.path.join(MODEL_DIR, f"{genre.lower()}_model.h5")
            seed_data_path = os.path.join(SEED_DATA_DIR, f"{genre.lower()}_seed.npy")
            
            if os.path.exists(model_path):
                print(f"✅ Model for {genre} already exists. Loading model...")
                model = load_model(model_path)
                
                # Check if seed data exists
                if not os.path.exists(seed_data_path):
                    # Generate and save seed data if not exists
                    X = prepare_data_from_folder(path, sequence_length)
                    generate_and_save_seed_data(X, genre)
                else:
                    print(f"✅ Seed data for {genre} already exists.")
            else:
                print(f"❌ Model for {genre} not found. Training a new model...")
                try:
                    train_and_save_model_for_genre(genre, path)
                except Exception as e:
                    print(f"❌ Error training {genre}: {e}")
        else:
            print(f"⚠️ Skipping {genre} - path does not exist: {path}")
