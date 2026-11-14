"""Turn generated tokens into playable files.

* ``tokens_to_midi``  -> a standard ``.mid`` file (PrettyMIDI), playable anywhere.
* ``midi_to_wav``     -> a ``.wav`` preview rendered by a small built-in synthesizer
  (numpy only), so the browser can play the result without FluidSynth or a soundfont.
  The preview timbre is piano-like for every instrument; the ``.mid`` file itself
  carries the instrument you picked.
"""
from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pretty_midi

from preprocess import decode_token

SAMPLE_RATE = 22050


def tokens_to_pretty_midi(
    tokens: Iterable[str],
    tempo: float = 100.0,
    program: int = 0,
    base_velocity: int = 80,
    humanize: bool = True,
    seed: Optional[int] = None,
) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI object. Malformed tokens are skipped, never fatal."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(program=int(program), name="Generated")
    seconds_per_beat = 60.0 / float(tempo)
    rng = random.Random(seed)

    cursor = 0.0
    for token in tokens:
        try:
            pitches, duration = decode_token(token)
        except ValueError:
            continue
        length = duration * seconds_per_beat
        for pitch in pitches:
            if not 0 <= pitch <= 127:
                continue
            velocity = base_velocity + (rng.randint(-8, 8) if humanize else 0)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(min(127, max(1, velocity))),
                    pitch=int(pitch),
                    start=cursor,
                    end=cursor + max(0.05, length * 0.96),  # tiny gap re-articulates repeats
                )
            )
        cursor += length
    midi.instruments.append(instrument)
    return midi


def tokens_to_midi(tokens: Iterable[str], path: Path | str, **kwargs) -> pretty_midi.PrettyMIDI:
    """Write tokens to ``path`` as a MIDI file. Raises ``ValueError`` if nothing playable."""
    midi = tokens_to_pretty_midi(tokens, **kwargs)
    if not midi.instruments[0].notes:
        raise ValueError("The generated sequence contained no playable notes.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return midi


def synthesize(midi: pretty_midi.PrettyMIDI, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Additive-synthesis preview: a few harmonics, fast attack, decaying envelope."""
    audio = np.zeros(int((midi.get_end_time() + 1.5) * sample_rate) + 1, dtype=np.float64)
    harmonics = ((1, 1.0), (2, 0.5), (3, 0.28), (4, 0.14), (5, 0.07))
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            freq = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
            length = max(note.end - note.start, 0.05)
            count = int((length + 0.4) * sample_rate)
            t = np.arange(count) / sample_rate
            tone = sum(a * np.sin(2 * np.pi * freq * h * t) for h, a in harmonics if freq * h < sample_rate / 2)
            envelope = (
                np.minimum(1.0, t / 0.01)                       # attack
                * np.exp(-t * (1.0 + freq / 700.0))             # decay (higher = shorter)
                * np.where(t > length, np.exp(-(t - length) * 14.0), 1.0)  # release
            )
            segment = (note.velocity / 127.0) * 0.25 * tone * envelope
            start = int(note.start * sample_rate)
            stop = min(len(audio), start + count)
            audio[start:stop] += segment[: stop - start]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = np.tanh(audio / peak * 1.4) * 0.9  # normalise + gentle soft clip
    return audio.astype(np.float32)


def write_wav(audio: np.ndarray, path: Path | str, sample_rate: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def midi_to_wav(midi: pretty_midi.PrettyMIDI, path: Path | str) -> Path:
    write_wav(synthesize(midi), path)
    return Path(path)
