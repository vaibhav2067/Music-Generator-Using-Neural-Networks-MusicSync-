from flask import Flask, render_template, request, jsonify, send_file
<<<<<<< Updated upstream
import os
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
=======
from flask_socketio import SocketIO, emit
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
import pretty_midi
import time
import os
from google.oauth2 import id_token
from google.auth.transport import requests
>>>>>>> Stashed changes

app = Flask(__name__)
<<<<<<< Updated upstream

# Parameters
=======
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'music_sync_db'
mysql = MySQL(app)

# -----------------------------------------------
# Login Manager Setup
# -----------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", [user_id])
    user = cur.fetchone()
    if user:
        return User(user[0], user[1], user[2], user[3])
    return None

# -----------------------------------------------
# User Model
# -----------------------------------------------
class User(UserMixin):
    def __init__(self, id, google_id, email, username):
        self.id = id
        self.google_id = google_id
        self.email = email
        self.username = username

    def get_id(self):
        return str(self.id)

# -----------------------------------------------
# Constants
# -----------------------------------------------
>>>>>>> Stashed changes
vocab_size = 128
sequence_length = 50
generated_length = 100
MODEL_FOLDER = "saved_models"
SEED_FOLDER = "saved_seed_data"
GENERATED_MUSIC_FOLDER = "new_generated_musics"

<<<<<<< Updated upstream
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
=======
os.makedirs(GENERATED_MUSIC_FOLDER, exist_ok=True)

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------
>>>>>>> Stashed changes
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

<<<<<<< Updated upstream
# Generate music
def generate_music(model, seed_sequence, length=generated_length, temperature=1.0):
    generated = seed_sequence
    for _ in range(length):
        prediction = model.predict(np.expand_dims(generated, axis=0))
        next_note = sample_with_temperature(prediction[0], temperature)
        generated = np.append(generated[1:], next_note)
=======
def generate_music(model, seed_sequence, length=generated_length, temperature=1.0, sid=None):
    generated = list(seed_sequence)
    for i in range(length):
        prediction = model.predict(np.expand_dims(generated[-(sequence_length - 1):], axis=0), verbose=0)
        next_note = sample_with_temperature(prediction[0], temperature)
        generated.append(next_note)

        if sid:
            progress = int((i + 1) / length * 100)
            socketio.emit('generation_progress', {'progress': progress}, room=sid)

>>>>>>> Stashed changes
    return generated

# Convert sequence to MIDI
def sequence_to_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
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

<<<<<<< Updated upstream
# Serve index.html
=======
# -----------------------------------------------
# Routes
# -----------------------------------------------

>>>>>>> Stashed changes
@app.route('/')
def index():
    return render_template('index.html')

# Route to generate music
@app.route('/generate-music', methods=['POST'])
def generate_music_route():
    try:
        genre = request.form['genre']
        temperature = float(request.form['temperature'])
        sid = request.form.get('sid')

        model_path = os.path.join(MODEL_FOLDER, f"{genre.lower()}_model.h5")
        seed_path = os.path.join(SEED_FOLDER, f"{genre.lower()}_seed.npy")

<<<<<<< Updated upstream
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
 
=======
        # Check if model and seed files exist
        if not os.path.exists(model_path) or not os.path.exists(seed_path):
            return jsonify(success=False, error="Model or seed data not found for selected genre.")

        # Load the trained model and seed data
        model = load_model(model_path)
        X_seed = np.load(seed_path)
        
        # Select a random seed sequence
        seed_sequence = X_seed[np.random.randint(0, len(X_seed))]

        # Generate music using the model
        generated_sequence = generate_music(model, seed_sequence, temperature=temperature, sid=sid)

        # Save generated music to a .mid file
        user_id = current_user.id if current_user.is_authenticated else "guest"
        timestamp = int(time.time())
        filename = f"{user_id}_{genre}_{timestamp}_music.mid"
        output_file = os.path.join(GENERATED_MUSIC_FOLDER, filename)

        sequence_to_midi(generated_sequence, output_file)

        return jsonify(success=True, fileUrl=f"/download/{filename}")
    
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(GENERATED_MUSIC_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

@app.route('/google-signin', methods=['POST'])
def google_signin():
    data = request.json
    token = data.get('credential')
    try:
        CLIENT_ID = "your_google_client_id"
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)

        user_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE google_id = %s", (user_id,))
        user = cur.fetchone()

        if not user:
            cur.execute("INSERT INTO users (google_id, email, username) VALUES (%s, %s, %s)", 
                        (user_id, email, name))
            mysql.connection.commit()

        return jsonify(success=True, user={"id": user_id, "email": email, "name": name})
    except ValueError:
        return jsonify(success=False), 400

# -----------------------------------------------
# Run Server
# -----------------------------------------------
>>>>>>> Stashed changes
if __name__ == '__main__':
    socketio.run(app, debug=True)
