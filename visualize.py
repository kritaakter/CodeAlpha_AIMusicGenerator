"""Matplotlib figures for the dashboard.

Styling follows a few simple rules: one y-axis per chart, thin marks, recessive
grid, text in neutral ink (never the series colour), and a legend whenever a
chart has more than one series. Train vs. validation also differ by line style,
so identity never relies on colour alone.
"""
from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit renders the figure itself
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from preprocess import decode_token

SURFACE = "#FFFFFF"
INK = "#1F2430"
MUTED = "#6B7280"
GRID = "#E5E7EB"
SERIES_1 = "#5B3FE0"   # violet
SERIES_2 = "#0E8F80"   # teal

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _figure(width: float, height: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_training_history(history: dict):
    """Loss and accuracy per epoch, train (solid) vs validation (dashed)."""
    fig, (ax_loss, ax_acc) = _figure(10, 3.4, 2)
    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "accuracy", "Next-token accuracy")):
        ax.plot(epochs, history.get(key, []), color=SERIES_1, linewidth=2, label="Train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], color=SERIES_2, linewidth=2, linestyle="--", label="Validation")
        _style(ax, title, "Epoch")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        if len(epochs) <= 12:
            ax.set_xticks(epochs)
    fig.tight_layout()
    return fig


def plot_piano_roll(tokens: list[str], tempo: float = 100.0):
    """Notes as bars: time (seconds) on x, MIDI pitch on y."""
    fig, ax = _figure(10, 3.6)
    seconds_per_beat = 60.0 / tempo
    cursor, lows, highs = 0.0, [], []
    for token in tokens:
        try:
            pitches, duration = decode_token(token)
        except ValueError:
            continue
        for pitch in pitches:
            ax.add_patch(
                Rectangle((cursor, pitch - 0.4), duration * seconds_per_beat * 0.96, 0.8,
                          facecolor=SERIES_1, edgecolor=SURFACE, linewidth=0.5)
            )
            lows.append(pitch)
            highs.append(pitch)
        cursor += duration * seconds_per_beat
    if lows:
        ax.set_xlim(0, max(cursor, 1))
        ax.set_ylim(min(lows) - 3, max(highs) + 3)
    _style(ax, "Piano roll", "Time (seconds)", "MIDI pitch")
    fig.tight_layout()
    return fig


def _pitch_class_counts(counts_128: np.ndarray) -> np.ndarray:
    pc = np.zeros(12)
    for pitch, count in enumerate(counts_128):
        pc[pitch % 12] += count
    return pc


def plot_pitch_classes(counts_128, title: str = "Pitch-class usage"):
    """Share of notes per pitch class (C, C#, ... B)."""
    pc = _pitch_class_counts(np.asarray(counts_128, dtype=float))
    share = pc / pc.sum() * 100 if pc.sum() else pc
    fig, ax = _figure(5, 3.2)
    ax.bar(NOTE_NAMES, share, color=SERIES_1, width=0.62)
    _style(ax, title, "", "% of notes")
    fig.tight_layout()
    return fig


def tokens_pitch_histogram(tokens: list[str]) -> np.ndarray:
    counts = np.zeros(128, dtype=np.int64)
    for token in tokens:
        try:
            for p in decode_token(token)[0]:
                counts[p] += 1
        except ValueError:
            continue
    return counts


def plot_duration_distribution(duration_counts: dict, title: str = "Note durations"):
    """Number of events per duration (in beats)."""
    items = sorted(((float(k), v) for k, v in duration_counts.items()))
    fig, ax = _figure(5, 3.2)
    ax.bar([f"{k:g}" for k, _ in items], [v for _, v in items], color=SERIES_2, width=0.62)
    _style(ax, title, "Duration (beats)", "Events")
    fig.tight_layout()
    return fig


def tokens_duration_counts(tokens: list[str]) -> dict:
    counter: Counter = Counter()
    for token in tokens:
        try:
            counter[decode_token(token)[1]] += 1
        except ValueError:
            continue
    return dict(counter)


def plot_pitch_range(counts_128, title: str = "Pitch distribution (MIDI 21-108)"):
    counts = np.asarray(counts_128)[21:109]
    fig, ax = _figure(10, 3.0)
    ax.bar(np.arange(21, 109), counts, color=SERIES_1, width=0.8)
    _style(ax, title, "MIDI pitch", "Notes")
    fig.tight_layout()
    return fig
