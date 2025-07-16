# -----------------------------------------------
# Imports
# -----------------------------------------------
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
import time
import os
import numpy as np
import pretty_midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.layers import Input, Concatenate
from google.oauth2 import id_token
from google.auth.transport import requests

# -----------------------------------------------
# Flask App Initialization
# -----------------------------------------------
app = Flask(__name__)   
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow CORS for SocketIO
app.config['SECRET_KEY'] = '2fbf06ef60cfd1fb4315dfde1323a510111ad118bdcfd6ba02d55747b09dfe4d'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Vais@9786'
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
# Constants & Configuration
# -----------------------------------------------
vocab_size = 128
sequence_length = 50
embedding_dim = 64
epochs = 1
batch_size = 64
generated_length = 100

genre_folders = {
    "Classic": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Classic",
    "Pop": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Pop",
    "Rock": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Rock",
    "Rap": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Rap",
    "Dance": "C:/Users/vaibh/Documents/GitHub/Music-Generator-Using-Neural-Networks/MIDI files (Genre)/Dance"
}

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------
def midi_to_sequence(midi_file, sequence_length):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None

    sequences = []
    for instrument in midi.instruments:
        instrument_id = instrument.program
        notes = [(note.pitch, note.start) for note in instrument.notes]

        if len(notes) < sequence_length:
            continue

        notes = np.array(notes)
        pitch_data = notes[:, 0] % vocab_size
        timing_data = np.diff(notes[:, 1], prepend=notes[0, 1])

        # instrument_arr = np.full(len(notes), instrument_id)

        combined = np.stack((pitch_data, timing_data, np.full(len(pitch_data), instrument_id)), axis=1)
        for i in range(len(pitch_data) - sequence_length):
            sequences.append(combined[i:i + sequence_length])

    return np.array(sequences) if sequences else None

def prepare_data_from_folder(folder_path, sequence_length):
    midi_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mid')]
    X = [midi_to_sequence(midi_file, sequence_length) for midi_file in midi_files if midi_to_sequence(midi_file, sequence_length) is not None]
    if not X:
        raise ValueError("No valid MIDI files found.")
    return np.concatenate([x for x in X if x.ndim == 2], axis=0)

def create_model():
    pitch_input = Input(shape=(sequence_length - 1,))
    timing_input = Input(shape=(sequence_length - 1,))
    instrument_input = Input(shape=(sequence_length - 1,))

    pitch_embedding = Embedding(vocab_size, embedding_dim)(pitch_input)
    instrument_embedding = Embedding(128, embedding_dim)(instrument_input)

    merged = Concatenate()([pitch_embedding, instrument_embedding, timing_input[..., None]])

    lstm = LSTM(128, return_sequences=True)(merged)
    lstm = LSTM(128)(lstm)
    
    pitch_output = Dense(vocab_size, activation='softmax', name='pitch_output')(lstm)
    timing_output = Dense(1, activation='relu', name='timing_output')(lstm)
    instrument_output = Dense(128, activation='softmax', name='instrument_output')(lstm)

    model = Model(inputs=[pitch_input, timing_input, instrument_input], outputs=[pitch_output, timing_output, instrument_output])
    
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', 'mse', 'sparse_categorical_crossentropy'])
    return model

def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)

def generate_music(model, seed_sequence, length=generated_length, temperature=1.0):
    generated_pitch = list(seed_sequence[:, 0])
    generated_timing = list(seed_sequence[:, 1])
    generated_instrument = list(seed_sequence[:, 2])

    for _ in range(length):
        prediction = model.predict([
            np.expand_dims(generated_pitch[-(sequence_length - 1):], axis=0),
            np.expand_dims(generated_timing[-(sequence_length - 1):], axis=0),
            np.expand_dims(generated_instrument[-(sequence_length - 1):], axis=0)
        ], verbose=0)

        next_pitch = sample_with_temperature(prediction[0][0], temperature)
        next_timing = prediction[1][0]
        next_instrument = np.argmax(prediction[2][0])  # Get most likely instrument

        generated_pitch.append(next_pitch)
        generated_timing.append(next_timing)
        generated_instrument.append(next_instrument)

    return np.stack([generated_pitch, generated_timing, generated_instrument], axis=1)

def sequence_to_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    instruments = {}

    start_time = 0
    for note_value, timing, instrument_id in sequence:
        if instrument_id not in instruments:
            instruments[instrument_id] = pretty_midi.Instrument(program=int(instrument_id))

        note = pretty_midi.Note(
            velocity=100,
            pitch=int(note_value),
            start=start_time,
            end=start_time + max(0.1, float(timing))  # Ensure minimum duration
        )

        instruments[instrument_id].notes.append(note)
        start_time += max(0.1, float(timing))  # Use timing for rhythm

    for inst in instruments.values():
        midi.instruments.append(inst)

    midi.write(output_file)

# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get-started')
def get_started():
    return render_template('index.html')

@app.route('/generate-music', methods=['POST'])
def generate_music_route():
    try:
        genre = request.form['genre']
        temperature = float(request.form['temperature'])
        folder_path = genre_folders.get(genre)

        if not folder_path:
            return jsonify(success=False, error="Invalid genre selected.")

        X_train = prepare_data_from_folder(folder_path, sequence_length)
        y_train = X_train[:, -1]
        X_train = X_train[:, :-1]

        model = create_model()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        seed_sequence = X_train[np.random.randint(0, len(X_train))]
        generated_sequence = generate_music(model, seed_sequence, temperature=temperature)
        user_id = current_user.id if current_user.is_authenticated else "guest"
        output_file = os.path.join("static", f"{user_id}_generated_music.mid")
        sequence_to_midi(generated_sequence, output_file)

        return jsonify(success=True, fileUrl=f"/download/{os.path.basename(output_file)}")

    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/cancel-generation', methods=['POST'])
def cancel_generation():
    global generation_cancelled
    generation_cancelled = True
    return jsonify(success=True)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('static', filename)
    return send_file(file_path, as_attachment=True)

@app.route('/google-signin', methods=['POST'])
def google_signin():
    data = request.json
    token = data.get('credential')
    try:
        CLIENT_ID = "77799141467-llp0bdb8q524cb4ml5eloqk8cvjhfpj7.apps.googleusercontent.com"
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)

        user_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (google_id, email, username) VALUES (%s, %s, %s)", 
                    (user_id, email, name))
        mysql.connection.commit()

        return jsonify(success=True, user={"id": user_id, "email": email, "name": name})
    except ValueError:
        return jsonify(success=False), 400

# -----------------------------------------------
# Main
# -----------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
