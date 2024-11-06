# Music Generator Application

This application is a Flask-based music generator that uses neural networks to create music sequences. Users can select a genre, input parameters, and receive a generated MIDI file as output. The model is trained on MIDI files for specified genres (e.g., Classic, Pop) and uses an LSTM-based neural network to predict note sequences.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Routes](#routes)
- [Contributing](#contributing)

## Features

- Generates MIDI music sequences based on genre-specific datasets.
- Allows user-defined temperature parameter for controlling creativity in generated music.
- Downloadable MIDI files of generated sequences.
- Simple web interface for selecting genre and temperature.

## Technologies Used

- **Flask**: For handling HTTP requests and serving HTML pages.
- **TensorFlow**: For building the LSTM model.
- **pretty_midi**: For MIDI file manipulation.
- **NumPy**: For data processing and array manipulation.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/music-generator-app.git
    cd music-generator-app
    ```

2. **Install dependencies**:
    Make sure you have Python installed, then run:
    ```bash
    pip install Flask numpy pretty_midi tensorflow
    ```

3. **Directory setup**:
    Place MIDI files for each genre in their respective folders (e.g., `Classic`, `Pop`). Update paths in the code if necessary.

## Usage

1. **Run the application**:
    ```bash
    python app.py
    ```

2. **Access the web interface**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

3. **Generate Music**:
   - Choose a genre and specify the temperature.
   - Click **Generate Music**.
   - A downloadable link to the generated MIDI file will appear.

## Project Structure

```plaintext
.
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # Frontend HTML page
├── static/               # Folder for static files
│   └── (CSS, JS, etc.)   
└── requirements.txt      # Python dependencies
```
## Routes

- GET /: Renders the home page where users can input generation parameters.
- POST /generate-music: Generates a MIDI music sequence based on selected genre and temperature.
- GET /download/<filename>: Serves the generated MIDI file for download.

## Contributing

If you want to contribute to this project, please fork the repository and create a pull request. Any enhancements to the model, dataset, or user interface are welcome.
