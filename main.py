# -----------------------------------------------
# Imports
# -----------------------------------------------
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from tensorflow.keras.models import load_model 
import numpy as np
import pretty_midi
import time
import os
from google.oauth2 import id_token
from google.auth.transport import requests

# -----------------------------------------------
# Flask App Initialization
# -----------------------------------------------
app = Flask(__name__)
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
vocab_size = 128
sequence_length = 50
generated_length = 100
MODEL_FOLDER = "saved_models"
SEED_FOLDER = "saved_seed_data"
GENERATED_MUSIC_FOLDER = "new_generated_musics"

os.makedirs(GENERATED_MUSIC_FOLDER, exist_ok=True)

# -----------------------------------------------
# Genre-Specific Instrument & Harmony Configs
# -----------------------------------------------
GENRE_INSTRUMENTS = {
    'jazz': [0, 33, 40],         # Piano, Electric Bass, Violin
    'rock': [29, 30, 35],        # Muted Guitar, Overdriven Guitar, Fretless Bass
    'classical': [0, 40, 41],    # Piano, Violin, Viola
    'hiphop': [0, 118, 25],      # Piano, Synth Drum, Acoustic Guitar
    'electronic': [81, 89, 90],  # Lead 1, Pad 2, Pad 3
}

HARMONY_INTERVALS = {
    'jazz': [0, 4, 7],         # Root, major third, perfect fifth
    'rock': [0, 7],            # Power chord (root + 5th)
    'classical': [0, 3, 7],    # Minor/major triad
    'hiphop': [0, 12],         # Root + octave
    'electronic': [0, 7, 12],  # Fifth + octave
}

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)

def generate_music(model, seed_sequence, length=generated_length, temperature=1.0, sid=None):
    generated = list(seed_sequence)
    for i in range(length):
        prediction = model.predict(np.expand_dims(generated[-(sequence_length - 1):], axis=0), verbose=0)
        next_note = sample_with_temperature(prediction[0], temperature)
        generated.append(next_note)

        if sid:
            progress = int((i + 1) / length * 100)
            socketio.emit('generation_progress', {'progress': progress}, room=sid)

    return generated

def sequence_to_midi(sequence, output_file, genre='default'):
    midi = pretty_midi.PrettyMIDI()
    instruments = GENRE_INSTRUMENTS.get(genre.lower(), [0])  # Get instruments based on genre
    intervals = HARMONY_INTERVALS.get(genre.lower(), [0])  # Get harmony intervals based on genre
    start_time = 0.0

    for note_value in sequence:
        for idx, program in enumerate(instruments):
            instrument = pretty_midi.Instrument(program=program)

            # Generate harmony by adding intervals to the note
            for interval in intervals:
                pitch = int(note_value) + interval
                if pitch >= 128:  # Avoid pitches that exceed MIDI range
                    continue

                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start_time,
                    end=start_time + 0.5  # You can adjust the note length here
                )
                instrument.notes.append(note)

            midi.instruments.append(instrument)
            break  # Use the first instrument for harmony (avoiding duplicate chords)

        start_time += 0.5  # Adjust the timing of each note (change the duration for rhythm complexity)

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
        sid = request.form.get('sid')

        model_path = os.path.join(MODEL_FOLDER, f"{genre.lower()}_model.h5")
        seed_path = os.path.join(SEED_FOLDER, f"{genre.lower()}_seed.npy")

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

        sequence_to_midi(generated_sequence, output_file, genre=genre)  # Pass genre to sequence_to_midi

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
if __name__ == '__main__':
    socketio.run(app, debug=True)
