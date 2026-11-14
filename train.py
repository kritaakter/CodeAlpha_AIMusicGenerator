"""Train the LSTM on the preprocessed MIDI dataset.

Run from the command line:   python train.py --epochs 30
or from the Streamlit app (Train tab), which passes ``on_epoch_end`` for live progress.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Callable, Optional

import numpy as np

from config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    EMBEDDING_DIM,
    EPOCHS,
    HISTORY_FILE,
    LEARNING_RATE,
    LSTM_UNITS,
    MIN_TRAIN_SEQUENCES,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_INFO_FILE,
    VALIDATION_SPLIT,
)
from preprocess import DatasetError, load_dataset

log = logging.getLogger(__name__)
EpochCB = Optional[Callable[[int, int, dict], None]]


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lstm_units: int = LSTM_UNITS,
    embedding_dim: int = EMBEDDING_DIM,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
    on_epoch_end: EpochCB = None,
) -> dict:
    """Train, save the best model plus history, and return a training-info dict."""
    import tensorflow as tf
    from tensorflow import keras

    from model import build_model

    X, y, _, vocab, piece = load_dataset(with_pieces=True)
    vocab_size = len(vocab["id_to_token"])
    if len(X) < MIN_TRAIN_SEQUENCES:
        raise DatasetError(f"Only {len(X)} sequences available; need at least {MIN_TRAIN_SEQUENCES}.")

    # Hold out whole *pieces* (chosen at random) for validation. Sliding windows
    # overlap heavily, so a random window split would leak near-duplicates.
    piece_ids = np.unique(piece)
    rng = np.random.default_rng(RANDOM_SEED)
    n_val_pieces = max(1, int(round(len(piece_ids) * VALIDATION_SPLIT))) if len(piece_ids) > 1 else 0
    val_pieces = rng.choice(piece_ids, size=n_val_pieces, replace=False) if n_val_pieces else []
    is_val = np.isin(piece, val_pieces)
    if is_val.sum() == 0:  # single-piece dataset: fall back to a tail split
        is_val[int(len(X) * (1 - VALIDATION_SPLIT)):] = True
    x_train, y_train, x_val, y_val = X[~is_val], y[~is_val], X[is_val], y[is_val]

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(
        vocab_size, X.shape[1], embedding_dim, lstm_units, dropout, learning_rate
    )

    class _Progress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch_end:
                on_epoch_end(epoch + 1, epochs, {k: float(v) for k, v in (logs or {}).items()})

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(MODEL_FILE), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        _Progress(),
    ]

    started = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    seconds = time.time() - started
    model.save(MODEL_FILE)  # best weights were restored by EarlyStopping

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    HISTORY_FILE.write_text(json.dumps(hist))

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    val_loss = hist["val_loss"]
    best_epoch = int(np.argmin(val_loss)) + 1
    info = {
        "epochs_requested": epochs,
        "epochs_run": len(val_loss),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(val_loss)),
        "best_val_accuracy": float(hist["val_accuracy"][best_epoch - 1]),
        "best_val_perplexity": float(np.exp(min(val_loss))),
        "final_train_loss": hist["loss"][-1],
        "final_train_accuracy": hist["accuracy"][-1],
        "batch_size": batch_size,
        "lstm_units": lstm_units,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "vocab_size": vocab_size,
        "sequence_length": int(X.shape[1]),
        "train_sequences": int(len(x_train)),
        "val_sequences": int(len(x_val)),
        "parameters": int(model.count_params()),
        "seconds": round(seconds, 1),
        "tensorflow": tf.__version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_summary": "\n".join(summary_lines),
    }
    TRAIN_INFO_FILE.write_text(json.dumps(info, indent=2))
    return info


def load_history() -> Optional[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (OSError, ValueError):
        return None


def load_training_info() -> Optional[dict]:
    try:
        return json.loads(TRAIN_INFO_FILE.read_text())
    except (OSError, ValueError):
        return None


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train the music LSTM.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--units", type=int, default=LSTM_UNITS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    def show(epoch, total, logs):
        print(
            f"epoch {epoch}/{total}  loss {logs.get('loss', 0):.3f}  acc {logs.get('accuracy', 0):.3f}"
            f"  val_loss {logs.get('val_loss', 0):.3f}  val_acc {logs.get('val_accuracy', 0):.3f}"
        )

    try:
        info = train_model(
            args.epochs, args.batch_size, args.units, learning_rate=args.learning_rate, on_epoch_end=show
        )
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}\nRun `python preprocess.py` first.")
    print(json.dumps({k: v for k, v in info.items() if k != "model_summary"}, indent=2))
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
