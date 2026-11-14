"""Preprocessing: MIDI -> music21 -> event tokens -> numeric training sequences.

Representation
--------------
An *event* = "play these pitches, then advance time by this many beats":

    "60.64.67|1"   C-major chord (MIDI pitches 60, 64, 67), held for 1 beat
    "72|0.5"       single note C5, half a beat

Events are split into *atomic tokens* for the network: ``n60 n64 n67 d1``
(note tokens in ascending order, closed by a duration token). That keeps the
vocabulary around 100 tokens, which a small dataset can actually cover.

Pieces are transposed to C major / A minor first, so the network learns one
tonal centre instead of twelve. Tokens are mapped to integer ids, and sliding
windows of ``SEQUENCE_LENGTH`` ids become the inputs; the id that follows each
window is the target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import (
    DATASET_FILE,
    DURATION_CHOICES,
    MAX_CHORD_NOTES,
    MAX_EVENTS_PER_FILE,
    MAX_FILES,
    MIDI_DIR,
    MIN_EVENTS_PER_FILE,
    MIN_TOKEN_FREQ,
    MIN_TRAIN_SEQUENCES,
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
    STATS_FILE,
    TRANSPOSE_TO_C,
    VOCAB_FILE,
    ensure_dirs,
)
from dataset import is_valid_midi_file, list_midi_files

log = logging.getLogger(__name__)
ProgressCB = Optional[Callable[[int, int, str], None]]

LOWEST_PITCH, HIGHEST_PITCH = 21, 108  # piano range


class DatasetError(Exception):
    """Raised when there is no usable dataset (missing, empty or too small)."""


# --------------------------------------------------------- token helpers
def encode_event(pitches: list[int], duration: float) -> str:
    return ".".join(str(int(p)) for p in pitches) + "|" + f"{duration:g}"


def decode_token(token: str) -> tuple[list[int], float]:
    """Inverse of :func:`encode_event`. Raises ``ValueError`` on malformed tokens."""
    try:
        pitch_part, duration_part = token.split("|")
        pitches = [int(p) for p in pitch_part.split(".") if p != ""]
        duration = float(duration_part)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid token: {token!r}") from exc
    if not pitches or duration <= 0:
        raise ValueError(f"Invalid token: {token!r}")
    return pitches, duration


def events_to_atoms(events: list[str]) -> list[str]:
    """Split events into atomic tokens: ``"60.64|1"`` -> ``["n60", "n64", "d1"]``.

    Atomic tokens keep the vocabulary tiny (~100) instead of one token per
    chord/duration combination (thousands), which a small dataset cannot cover.
    """
    atoms: list[str] = []
    for event in events:
        pitches, duration = decode_token(event)
        atoms.extend(f"n{p}" for p in sorted(pitches))
        atoms.append(f"d{duration:g}")
    return atoms


def atoms_to_events(atoms: list[str]) -> list[str]:
    """Inverse of :func:`events_to_atoms`; tolerant of malformed model output.

    Pitches accumulate until a duration atom closes the event. A duration with
    no pitches before it is dropped; trailing pitches with no duration are dropped.
    """
    events: list[str] = []
    pitches: list[int] = []
    for atom in atoms:
        try:
            if atom.startswith("n"):
                pitch = int(atom[1:])
                if pitch not in pitches:
                    pitches.append(pitch)
            elif atom.startswith("d") and pitches:
                events.append(encode_event(sorted(pitches), float(atom[1:])))
                pitches = []
        except ValueError:
            continue
    return events


def _snap_duration(value: float) -> float:
    capped = min(value, max(DURATION_CHOICES))
    return min(DURATION_CHOICES, key=lambda c: abs(c - capped))


def _key_shift(score) -> int:
    """Semitone shift that moves the detected key to C major / A minor."""
    try:
        key = score.analyze("key")
        target = 0 if key.mode == "major" else 9
        shift = (target - key.tonic.pitchClass) % 12
        return shift - 12 if shift > 6 else shift
    except Exception:
        return 0  # key analysis can fail on odd files; leave the piece as-is


# ------------------------------------------------------------ extraction
def extract_events(path: Path | str) -> list[str]:
    """Parse one MIDI file with music21 and return its list of event tokens.

    Raises on unreadable files -- the caller records the reason and moves on.
    """
    from music21 import converter  # heavy import, keep it lazy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = converter.parse(str(path))
        shift = _key_shift(score) if TRANSPOSE_TO_C else 0
        elements = list(score.flatten().notes)  # Notes and Chords, no rests

    # Group everything that starts at the same (16th-note quantised) offset.
    groups: dict[float, set[int]] = {}
    for element in elements:
        onset = round(float(element.offset) * 4) / 4
        midi_pitches = [p.midi for p in element.pitches] if element.isChord else [element.pitch.midi]
        groups.setdefault(onset, set()).update(int(p) for p in midi_pitches)

    onsets = sorted(groups)
    events: list[str] = []
    for i, onset in enumerate(onsets):
        pitches = sorted(p + shift for p in groups[onset])
        pitches = [p for p in pitches if LOWEST_PITCH <= p <= HIGHEST_PITCH]
        if not pitches:
            continue
        if len(pitches) > MAX_CHORD_NOTES:  # keep the bass plus the top voices
            pitches = [pitches[0]] + pitches[-(MAX_CHORD_NOTES - 1):]
        gap = (onsets[i + 1] - onset) if i + 1 < len(onsets) else 1.0
        events.append(encode_event(pitches, _snap_duration(gap)))
        if len(events) >= MAX_EVENTS_PER_FILE:
            break
    return events


def _cached_events(path: Path) -> list[str]:
    """``extract_events`` with an on-disk cache (music21 parsing is the slow part).

    The cache key includes the file's size/mtime and the transpose setting, so
    edited files and changed settings are re-parsed automatically.
    """
    stat = path.stat()
    key = hashlib.md5(
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{TRANSPOSE_TO_C}|{MAX_CHORD_NOTES}|{MAX_EVENTS_PER_FILE}".encode()
    ).hexdigest()
    cache_file = PROCESSED_DIR / "events_cache" / f"{key}.json"
    try:
        return json.loads(cache_file.read_text())
    except (OSError, ValueError):
        pass
    events = extract_events(path)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(events))
    except OSError:
        pass  # caching is best-effort
    return events


# --------------------------------------------------------- dataset build
def build_dataset(
    midi_dir: Path = MIDI_DIR,
    sequence_length: int = SEQUENCE_LENGTH,
    max_files: int = MAX_FILES,
    progress: ProgressCB = None,
) -> dict:
    """Parse MIDI files, build the vocabulary and save training arrays to disk."""
    ensure_dirs()
    started = time.time()
    files = list_midi_files(midi_dir)
    if not files:
        raise DatasetError(
            f"No MIDI files found in '{midi_dir}'. Add .mid files, or use the "
            "'Export classical pieces' option to create a dataset automatically."
        )
    found = len(files)
    if found > max_files:  # evenly spaced subset
        step = found / max_files
        files = [files[int(i * step)] for i in range(max_files)]

    per_file: list[tuple[str, list[str]]] = []
    skipped: list[dict] = []
    for i, path in enumerate(files, start=1):
        if progress:
            progress(i, len(files), path.name)
        if not is_valid_midi_file(path):
            skipped.append({"file": path.name, "reason": "invalid or corrupt MIDI header"})
            continue
        try:
            events = _cached_events(path)
        except Exception as exc:
            skipped.append({"file": path.name, "reason": f"parse error: {str(exc)[:90]}"})
            continue
        if len(events) < MIN_EVENTS_PER_FILE:
            skipped.append({"file": path.name, "reason": f"too short ({len(events)} events)"})
            continue
        per_file.append((path.name, events))

    if not per_file:
        raise DatasetError(
            "None of the MIDI files could be used (all were invalid, unreadable or too short). "
            f"First problems: {skipped[:3]}"
        )

    # Vocabulary: keep tokens that occur often enough to be learnable.
    atoms_per_file = [events_to_atoms(events) for _, events in per_file]
    counts = Counter(tok for atoms in atoms_per_file for tok in atoms)
    kept = [tok for tok, c in counts.most_common() if c >= MIN_TOKEN_FREQ]
    if len(kept) < 10:
        raise DatasetError(f"Vocabulary too small ({len(kept)} tokens). Add more MIDI files.")
    token_to_id = {tok: i for i, tok in enumerate(kept)}

    x_parts, y_parts, corpus_parts, piece_parts = [], [], [], []
    total_events = total_tokens = 0
    pitch_hist = np.zeros(128, dtype=np.int64)
    duration_counts: Counter = Counter()
    for (_, events), atoms in zip(per_file, atoms_per_file):
        ids = np.array([token_to_id[t] for t in atoms if t in token_to_id], dtype=np.int32)
        if len(ids) <= sequence_length:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(ids, sequence_length + 1)
        x_parts.append(windows[:, :-1])
        y_parts.append(windows[:, -1])
        piece_parts.append(np.full(len(windows), len(piece_parts), dtype=np.int32))  # piece id per window
        corpus_parts.extend([ids, np.array([-1], dtype=np.int32)])  # -1 separates pieces
        total_events += len(events)
        total_tokens += len(ids)
        for tok in events:
            pitches, duration = decode_token(tok)
            for p in pitches:
                pitch_hist[p] += 1
            duration_counts[duration] += 1

    if not x_parts:
        raise DatasetError("Files were too short after filtering rare tokens. Add more MIDI files.")
    X = np.concatenate(x_parts).astype(np.int32)
    y = np.concatenate(y_parts).astype(np.int32)
    if len(X) < MIN_TRAIN_SEQUENCES:
        raise DatasetError(
            f"Only {len(X)} training sequences were produced (need {MIN_TRAIN_SEQUENCES}+). "
            "Add more or longer MIDI files."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET_FILE, X=X, y=y, corpus=np.concatenate(corpus_parts), piece=np.concatenate(piece_parts)
    )
    VOCAB_FILE.write_text(
        json.dumps({"id_to_token": kept, "sequence_length": sequence_length}, ensure_ascii=False)
    )
    stats = {
        "files_found": found,
        "files_parsed": len(files),
        "files_used": len(per_file),
        "files_skipped": len(skipped),
        "skipped": skipped[:100],
        "total_events": int(total_events),
        "total_tokens": int(total_tokens),
        "vocab_size": len(kept),
        "n_sequences": int(len(X)),
        "sequence_length": sequence_length,
        "pitch_histogram": pitch_hist.tolist(),
        "duration_counts": {f"{k:g}": int(v) for k, v in sorted(duration_counts.items())},
        "top_tokens": [{"token": t, "count": int(counts[t])} for t in kept[:10]],
        "seconds": round(time.time() - started, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    return stats


# ------------------------------------------------------------- loading
def dataset_ready() -> bool:
    return DATASET_FILE.exists() and VOCAB_FILE.exists()


def load_dataset(with_pieces: bool = False):
    """Return ``(X, y, corpus, vocab)`` (plus ``piece`` ids if asked) or raise :class:`DatasetError`."""
    if not dataset_ready():
        raise DatasetError("Processed dataset not found. Run preprocessing first.")
    try:
        with np.load(DATASET_FILE) as data:
            X, y, corpus = data["X"], data["y"], data["corpus"]
            piece = data["piece"] if "piece" in data.files else np.zeros(len(X), dtype=np.int32)
        vocab = json.loads(VOCAB_FILE.read_text())
    except Exception as exc:
        raise DatasetError(f"Processed dataset is unreadable ({exc}). Re-run preprocessing.") from exc
    return (X, y, corpus, vocab, piece) if with_pieces else (X, y, corpus, vocab)


def load_stats() -> Optional[dict]:
    try:
        return json.loads(STATS_FILE.read_text())
    except (OSError, ValueError):
        return None


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MIDI files into training sequences.")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    args = parser.parse_args()
    try:
        stats = build_dataset(
            sequence_length=args.sequence_length,
            max_files=args.max_files,
            progress=lambda i, n, f: print(f"[{i}/{n}] {f}"),
        )
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("pitch_histogram", "skipped")}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
