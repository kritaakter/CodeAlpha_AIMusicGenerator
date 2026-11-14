# 🎼 AI Music Generator (LSTM)

Generate original classical-style music with an **LSTM neural network trained from scratch** on MIDI files.
No pretrained music model is used anywhere.

**Stack:** Python · TensorFlow/Keras · music21 · PrettyMIDI · NumPy · Matplotlib · Streamlit

## How it works

```
MIDI files ─▶ music21 ─▶ event tokens ─▶ integer sequences ─▶ LSTM ─▶ sampling ─▶ PrettyMIDI ─▶ .mid / .wav
 (dataset.py) (preprocess.py)            (preprocess.py)     (model.py, (generate.py) (midi_utils.py)
                                                              train.py)
```

* **Events**: music21 turns each MIDI file into events `pitches|duration`, e.g. `60.64.67|1` (C major chord,
  1 beat) or `72|0.5`. Pieces are transposed to C major / A minor so the model learns a single tonal centre.
* **Tokens**: each event is split into atomic tokens `n60 n64 n67 d1` (notes, then a duration). This keeps the
  vocabulary at ~100 tokens; treating every chord+duration combination as one token gives thousands of rare tokens
  that a small dataset cannot cover (the model overfits immediately).
* **Model**: `Embedding(64) → LSTM(256) → Dropout → LSTM(256) → Dropout → Dense(256) → Softmax(vocab)`.
  It reads 96 tokens (about 24 chords/notes) and predicts the next token.
* **Generation**: a random seed window from the training data, then temperature + top-k sampling until the
  requested number of events exists. The seed is discarded, so only *new* events are written to MIDI. A
  **novelty** score reports the share of short phrases (16 tokens ≈ 4 events) that never occur in the training data.
* Parsed events are cached in `data/processed/events_cache/`, so re-preprocessing is fast.

## Folder structure

```
ai-music-generator/
├── app.py            # Streamlit UI: dataset, train, generate, play, download
├── config.py         # Paths and hyperparameters
├── dataset.py        # Collect MIDI: music21 corpus export, zip download, uploads, validation
├── preprocess.py     # music21 parsing → tokens → numeric sequences
├── model.py          # Keras LSTM model
├── train.py          # Training loop, callbacks, saved history/metrics
├── generate.py       # Sampling, novelty metric, MIDI creation
├── midi_utils.py     # tokens → PrettyMIDI (.mid) and a built-in .wav preview synth
├── visualize.py      # Matplotlib charts
├── requirements.txt
├── data/midi/        # raw MIDI files (your dataset)
├── data/processed/   # generated: dataset.npz, vocab.json, stats.json
├── models/           # generated: music_lstm.keras, history.json, training_info.json
└── outputs/          # generated: .mid and .wav files
```

## Setup

Requires **Python 3.10 – 3.12** (3.13 also worked in testing with recent TensorFlow).

```bash
cd ai-music-generator
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### Option A – web app (recommended)

```bash
streamlit run app.py
```

1. **Dataset tab** – click *Export classical pieces* (Bach, Mozart, Beethoven, Haydn, Handel, Schumann;
   works offline), and/or upload your own `.mid` files, and/or paste a `.zip` URL. Then click *Preprocess dataset*.
2. **Train tab** – choose epochs and size, click *Start training*. Live loss/accuracy curves appear
   while it trains; the final metrics and model summary stay on the page.
3. **Generate tab** – set length, creativity (temperature), tempo and instrument, then *Generate music*.
   Play it in the browser and download the **.mid** (or .wav). A piano roll and statistics are shown.

### Option B – command line

```bash
python dataset.py --export-corpus --per-composer 20     # collect classical MIDI
python preprocess.py                                    # music21 → sequences
python train.py --epochs 30                             # train the LSTM
python generate.py --events 160 --temperature 0.9       # writes outputs/generated_*.mid
```

## Using your own dataset

Copy classical `.mid` / `.midi` files (e.g. from the free *MAESTRO*, *Classical Archives* or *Mutopia*
collections, subject to their licences) into `data/midi/` – subfolders are fine – and re-run preprocessing.
A few hundred pieces gives noticeably better music than a few dozen.

## Error handling

| Situation | What happens |
|---|---|
| `data/midi/` empty or missing | Clear message + one-click export of the music21 corpus |
| Corrupt / non-MIDI / too-short file | Skipped, reason listed in the *skipped files* table; the run continues |
| Too little usable data | Preprocessing stops with an explanation instead of training on noise |
| No trained model yet | Generate tab asks you to train first |
| Model and dataset out of sync (re-preprocessed after training) | Detected, asks you to retrain |
| Invalid upload or zip | Rejected with a specific message |

## Training time (important)

With the default settings (LSTM 256, ~100k training sequences) one epoch takes roughly **3 minutes on a 2-core
CPU**, so 30 epochs is a long run; a GPU makes it several times faster. For a quick first result use
**LSTM units 128, batch size 256, 5-10 epochs** (about 1-2 minutes per epoch on a 2-core CPU):

```bash
python train.py --epochs 8 --units 128 --batch-size 256 --learning-rate 0.002
```

Reference run on 59 classical pieces (Bach, Beethoven, Haydn, Mozart, Schumann ...): 5 epochs with that
setting reached 30.9 % validation next-token accuracy (loss 2.82 → 2.58, still improving), enough to produce
playable, key-consistent but loosely structured music. Longer training and a larger dataset improve it.

## Tuning tips

* **Repetitive output?** raise temperature (1.0–1.2) or top-k.
* **Chaotic output?** lower temperature (0.6–0.8) or train for more epochs on more data.
* **Slow training?** reduce LSTM units to 128 or parse fewer files. A GPU is used automatically if TensorFlow can see one.

## Notes

* The in-browser player uses a small built-in additive synthesizer so no soundfont or FluidSynth is needed.
  The downloaded `.mid` carries the instrument you selected and sounds richer in any MIDI player
  (e.g. MuseScore, GarageBand, VLC with a soundfont, or an online MIDI player).
* Validation holds out ~10 % of whole **pieces** (random, seeded) rather than random windows, because overlapping
  windows from the same piece would otherwise leak into validation. Validation accuracy on unseen pieces is
  naturally low for music (it is a hard, high-entropy prediction task) - watch validation *loss* instead.
