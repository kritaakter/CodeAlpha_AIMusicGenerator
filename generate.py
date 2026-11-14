"""Generate new music with the trained LSTM.

Process: pick a random seed window from the training data, repeatedly ask the
network for the next-token distribution, sample from it (temperature + top-k),
slide the window, and finally write the *new* tokens (not the seed) to MIDI.
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

from config import (
    DEFAULT_EVENTS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPO,
    DEFAULT_TOP_K,
    MODEL_FILE,
    NOVELTY_NGRAM,
    OUTPUT_DIR,
)
from midi_utils import midi_to_wav, tokens_to_midi
from preprocess import DatasetError, atoms_to_events, dataset_ready, decode_token, load_dataset


class ModelNotFoundError(Exception):
    """Raised when no trained model exists yet."""


def load_generator_artifacts() -> dict:
    """Load model + dataset pieces needed for generation, with clear errors."""
    if not MODEL_FILE.exists():
        raise ModelNotFoundError("No trained model found. Train the model first (Train tab or `python train.py`).")
    if not dataset_ready():
        raise DatasetError("Processed dataset missing. Run preprocessing first.")

    from tensorflow import keras

    try:
        model = keras.models.load_model(MODEL_FILE)
    except Exception as exc:
        raise ModelNotFoundError(f"Could not load the saved model ({exc}). Retrain it.") from exc

    X, _, corpus, vocab = load_dataset()
    id_to_token = vocab["id_to_token"]
    if model.output_shape[-1] != len(id_to_token) or model.input_shape[1] != X.shape[1]:
        raise ModelNotFoundError(
            "The saved model does not match the processed dataset (it was preprocessed again "
            "after training). Retrain the model."
        )
    # A compiled graph is ~50x faster than calling the model eagerly for every token.
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, int(X.shape[1])], tf.int32)])
    def predict_next(window):
        return model(window, training=False)

    return {
        "model": model,
        "predict": predict_next,
        "id_to_token": id_to_token,
        "X": X,
        "corpus": corpus,
        "ngrams": None,
    }


def _sample(probs: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    """Temperature + top-k sampling from a probability vector."""
    logits = np.log(np.asarray(probs, dtype=np.float64) + 1e-9) / max(temperature, 1e-3)
    if 0 < top_k < len(logits):
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_token_ids(
    artifacts: dict,
    n_events: int = DEFAULT_EVENTS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    rng_seed: Optional[int] = None,
) -> list[int]:
    """Autoregressively generate token ids until ``n_events`` musical events exist.

    An event is closed by a duration token (``d...``), so we count those. The
    seed window is not included in the result.
    """
    rng = np.random.default_rng(rng_seed)
    predict, X, id_to_token = artifacts["predict"], artifacts["X"], artifacts["id_to_token"]
    window = X[int(rng.integers(len(X)))].astype(np.int32).copy()

    generated: list[int] = []
    pending_notes = events_done = 0
    for _ in range(n_events * 8):  # safety cap against runs of duration-only output
        probs = predict(window[None, :]).numpy()[0]
        step_temp = temperature
        if len(generated) >= 16 and len(set(generated[-16:])) <= 3:
            step_temp *= 1.4  # break out of repetitive loops
        next_id = _sample(probs, step_temp, top_k, rng)
        atom = id_to_token[next_id]
        if atom.startswith("d") and pending_notes == 0:
            # A duration with no notes before it would make no sound; resample once.
            next_id = _sample(probs, step_temp, top_k, rng)
            atom = id_to_token[next_id]
        generated.append(next_id)
        window = np.append(window[1:], next_id).astype(np.int32)
        if atom.startswith("n"):
            pending_notes += 1
        elif pending_notes:
            pending_notes, events_done = 0, events_done + 1
            if events_done >= n_events:
                break
    return generated


def novelty_score(artifacts: dict, generated_ids: list[int], n: int = NOVELTY_NGRAM) -> float:
    """Fraction of generated n-grams that never occur in the training data (0..1)."""
    if artifacts.get("ngrams") is None:
        grams: set[tuple] = set()
        piece: list[int] = []
        for value in artifacts["corpus"].tolist() + [-1]:
            if value == -1:
                grams.update(tuple(piece[i:i + n]) for i in range(len(piece) - n + 1))
                piece = []
            else:
                piece.append(value)
        artifacts["ngrams"] = grams
    seen = artifacts["ngrams"]
    total = len(generated_ids) - n + 1
    if total <= 0:
        return 1.0
    fresh = sum(tuple(generated_ids[i:i + n]) not in seen for i in range(total))
    return fresh / total


def generate_music(
    artifacts: dict,
    n_events: int = DEFAULT_EVENTS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    tempo: float = DEFAULT_TEMPO,
    program: int = 0,
    rng_seed: Optional[int] = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Generate a piece, save ``.mid`` + ``.wav`` and return paths and statistics."""
    started = time.time()
    ids = generate_token_ids(artifacts, n_events, temperature, top_k, rng_seed)
    tokens = atoms_to_events([artifacts["id_to_token"][i] for i in ids])  # -> "60.64|1" events
    if not tokens:
        raise ValueError("The model produced no playable events. Train longer or raise the temperature.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"generated_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    midi_path = output_dir / f"{stem}.mid"
    midi = tokens_to_midi(tokens, midi_path, tempo=tempo, program=program, seed=rng_seed)

    # Re-open the file to prove it is a valid, playable MIDI.
    check = pretty_midi.PrettyMIDI(str(midi_path))
    n_notes = sum(len(i.notes) for i in check.instruments)
    if n_notes == 0:
        raise ValueError("Generated MIDI file is empty.")
    wav_path = midi_to_wav(check, output_dir / f"{stem}.wav")

    pitches = [p for t in tokens for p in decode_token(t)[0]]
    return {
        "tokens": tokens,
        "midi_path": str(midi_path),
        "wav_path": str(wav_path),
        "stats": {
            "events": len(tokens),
            "notes": n_notes,
            "duration_seconds": round(float(check.get_end_time()), 1),
            "unique_pitches": len(set(pitches)),
            "pitch_range": f"{min(pitches)}-{max(pitches)}",
            "novelty": round(novelty_score(artifacts, ids), 3),
            "temperature": temperature,
            "top_k": top_k,
            "tempo": tempo,
            "generation_seconds": round(time.time() - started, 1),
        },
        "tempo": tempo,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a MIDI file with the trained LSTM.")
    parser.add_argument("--events", type=int, default=DEFAULT_EVENTS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tempo", type=float, default=DEFAULT_TEMPO)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    try:
        art = load_generator_artifacts()
        result = generate_music(art, args.events, args.temperature, args.top_k, args.tempo, rng_seed=args.seed)
    except (ModelNotFoundError, DatasetError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"MIDI: {result['midi_path']}\nWAV : {result['wav_path']}\n{result['stats']}")


if __name__ == "__main__":
    _cli()
