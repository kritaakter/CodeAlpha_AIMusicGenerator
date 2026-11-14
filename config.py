"""Central configuration for the AI Music Generator.

Every path and hyperparameter lives here so the other modules stay free of
magic numbers. Values can be overridden from the Streamlit UI or the CLI.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MIDI_DIR = DATA_DIR / "midi"              # raw .mid files (your dataset)
PROCESSED_DIR = DATA_DIR / "processed"    # numeric sequences, vocab, stats
MODELS_DIR = BASE_DIR / "models"          # trained model + training logs
OUTPUT_DIR = BASE_DIR / "outputs"         # generated .mid / .wav files

DATASET_FILE = PROCESSED_DIR / "dataset.npz"
VOCAB_FILE = PROCESSED_DIR / "vocab.json"
STATS_FILE = PROCESSED_DIR / "stats.json"
MODEL_FILE = MODELS_DIR / "music_lstm.keras"
HISTORY_FILE = MODELS_DIR / "history.json"
TRAIN_INFO_FILE = MODELS_DIR / "training_info.json"

# -------------------------------------------------------- preprocessing
SEQUENCE_LENGTH = 96          # tokens (~24 chords/notes) the LSTM sees before predicting the next
MAX_CHORD_NOTES = 4           # chords are reduced to at most this many pitches
# Allowed durations in quarter-note beats. Every event is snapped to one.
DURATION_CHOICES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
MIN_TOKEN_FREQ = 3            # tokens rarer than this are dropped from the vocab
MAX_EVENTS_PER_FILE = 3000    # keeps huge pieces from dominating the dataset
MIN_EVENTS_PER_FILE = 48      # shorter files are skipped
MAX_FILES = 300               # cap on MIDI files parsed per preprocessing run
TRANSPOSE_TO_C = True         # transpose every piece to C major / A minor
MIN_TRAIN_SEQUENCES = 200     # refuse to train on less than this

# --------------------------------------------------------------- model
EMBEDDING_DIM = 64
LSTM_UNITS = 256
DROPOUT = 0.3
LEARNING_RATE = 1e-3

# ------------------------------------------------------------ training
EPOCHS = 30
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1        # held-out *tail* of the data (no window leakage)
RANDOM_SEED = 42
EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------- generation
DEFAULT_TEMPO = 100
DEFAULT_EVENTS = 128
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 12
NOVELTY_NGRAM = 16            # token n-gram (~4 events) used to measure originality

# General-MIDI programs offered in the UI (name -> program number)
INSTRUMENTS = {
    "Acoustic Grand Piano": 0,
    "Harpsichord": 6,
    "Church Organ": 19,
    "Acoustic Guitar (nylon)": 24,
    "Violin": 40,
    "Cello": 42,
    "String Ensemble": 48,
    "Flute": 73,
}

DEFAULT_COMPOSERS = ("bach", "mozart", "beethoven", "haydn", "handel", "schumann_robert")


def ensure_dirs() -> None:
    """Create every working directory if it does not exist yet."""
    for directory in (MIDI_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
