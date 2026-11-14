"""The LSTM network (TensorFlow / Keras), trained from scratch -- no pretrained weights."""
from __future__ import annotations

from config import DROPOUT, EMBEDDING_DIM, LEARNING_RATE, LSTM_UNITS, SEQUENCE_LENGTH


def build_model(
    vocab_size: int,
    sequence_length: int = SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
):
    """Embedding -> LSTM -> LSTM -> Dense -> softmax over the token vocabulary.

    Input : ``sequence_length`` token ids
    Output: probability distribution over the next token
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(sequence_length,), dtype="int32"),
            layers.Embedding(vocab_size, embedding_dim, name="token_embedding"),
            layers.LSTM(lstm_units, return_sequences=True, name="lstm_1"),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units, name="lstm_2"),
            layers.Dropout(dropout),
            layers.Dense(lstm_units, activation="relu", name="dense_hidden"),
            layers.Dense(vocab_size, activation="softmax", name="next_token"),
        ],
        name="music_lstm",
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
