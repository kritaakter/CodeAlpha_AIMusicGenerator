"""Dataset collection: find, validate, export and download classical MIDI files.

Three ways to get data into ``data/midi/``:
  1. Drop your own ``.mid`` / ``.midi`` files into the folder (or upload in the UI).
  2. Export classical pieces (Bach, Mozart, Beethoven ...) from the corpus that
     ships with music21 -- works offline.
  3. Download a ``.zip`` of MIDI files from a URL you provide.
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from config import DEFAULT_COMPOSERS, MIDI_DIR, ensure_dirs

log = logging.getLogger(__name__)

MIDI_EXTENSIONS = {".mid", ".midi"}
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024
ProgressCB = Optional[Callable[[int, int, str], None]]


# ------------------------------------------------------------ discovery
def list_midi_files(midi_dir: Path = MIDI_DIR) -> list[Path]:
    """All MIDI files below ``midi_dir`` (recursive, sorted). Empty if missing."""
    midi_dir = Path(midi_dir)
    if not midi_dir.exists():
        return []
    return sorted(
        p for p in midi_dir.rglob("*") if p.is_file() and p.suffix.lower() in MIDI_EXTENSIONS
    )


def is_valid_midi_file(path: Path | str) -> bool:
    """Cheap sanity check: file exists, is not tiny, and starts with the MIDI header."""
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size < 32:
            return False
        with open(p, "rb") as handle:
            return handle.read(4) == b"MThd"
    except OSError:
        return False


def _safe_name(name: str) -> str:
    """Strip directories and odd characters from a file name."""
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_") or "midi"
    return stem[:80]


# --------------------------------------------------------------- uploads
def save_uploaded_midi(name: str, data: bytes, midi_dir: Path = MIDI_DIR) -> tuple[bool, str]:
    """Validate and store one uploaded file. Returns ``(ok, message)``."""
    if Path(name).suffix.lower() not in MIDI_EXTENSIONS:
        return False, f"{name}: not a .mid/.midi file"
    if len(data) < 32 or data[:4] != b"MThd":
        return False, f"{name}: invalid MIDI header (file is corrupt or not MIDI)"
    target_dir = Path(midi_dir) / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_safe_name(name)}.mid"
    target.write_bytes(data)
    return True, f"{name}: saved"


# ---------------------------------------------------- music21 corpus export
def export_music21_corpus(
    composers: Iterable[str] = DEFAULT_COMPOSERS,
    max_per_composer: int = 15,
    midi_dir: Path = MIDI_DIR,
    progress: ProgressCB = None,
) -> dict:
    """Write classical pieces from music21's bundled corpus as MIDI files.

    Pieces are sampled evenly across each composer's catalogue so you get
    variety instead of 15 near-identical chorales in a row.
    """
    from music21 import corpus  # heavy import, keep it lazy

    out_dir = Path(midi_dir) / "music21_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, object]] = []
    errors: list[str] = []
    for composer in composers:
        try:
            paths = list(corpus.getComposer(composer))
        except Exception as exc:  # unknown composer name, corpus problem ...
            errors.append(f"{composer}: {exc}")
            continue
        if not paths:
            errors.append(f"{composer}: no pieces found in corpus")
            continue
        step = max(1, len(paths) // max(1, max_per_composer))
        for path in paths[::step][:max_per_composer]:
            plan.append((composer, path))

    written = skipped_existing = failed = 0
    for i, (composer, path) in enumerate(plan, start=1):
        stem = _safe_name(Path(str(path)).name.split(".")[0] + "_" + Path(str(path)).stem)
        target = out_dir / f"{composer}_{stem}.mid"
        if progress:
            progress(i, len(plan), target.name)
        if is_valid_midi_file(target):
            skipped_existing += 1
            continue
        try:
            score = corpus.parse(path)
            score.write("midi", fp=str(target))
            if is_valid_midi_file(target):
                written += 1
            else:
                failed += 1
                target.unlink(missing_ok=True)
        except Exception as exc:
            failed += 1
            errors.append(f"{Path(str(path)).name}: {str(exc)[:100]}")
            target.unlink(missing_ok=True)

    return {
        "requested": len(plan),
        "written": written,
        "already_present": skipped_existing,
        "failed": failed,
        "errors": errors[:20],
        "folder": str(out_dir),
    }


# ------------------------------------------------------------ zip download
def download_midi_zip(url: str, midi_dir: Path = MIDI_DIR, progress: ProgressCB = None) -> dict:
    """Download a ``.zip`` of MIDI files and keep only the valid ``.mid`` entries."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    request = urllib.request.Request(url, headers={"User-Agent": "ai-music-generator/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("Download is larger than the 300 MB safety limit.")

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded file is not a valid .zip archive.") from exc

    out_dir = Path(midi_dir) / "downloaded"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [i for i in archive.infolist() if Path(i.filename).suffix.lower() in MIDI_EXTENSIONS]
    saved = invalid = 0
    for n, info in enumerate(entries, start=1):
        if progress:
            progress(n, len(entries), info.filename)
        data = archive.read(info)  # we only use the *name*, never extract paths
        if len(data) < 32 or data[:4] != b"MThd":
            invalid += 1
            continue
        (out_dir / f"{_safe_name(info.filename)}_{n}.mid").write_bytes(data)
        saved += 1
    return {"midi_entries": len(entries), "saved": saved, "invalid": invalid, "folder": str(out_dir)}


# ---------------------------------------------------------------- helpers
def ensure_dataset(min_files: int = 1, export_if_missing: bool = True) -> dict:
    """Make sure some MIDI files exist; export the music21 corpus if the folder is empty."""
    ensure_dirs()
    files = list_midi_files()
    info: dict = {"found": len(files), "exported": None}
    if len(files) < min_files and export_if_missing:
        info["exported"] = export_music21_corpus()
        info["found"] = len(list_midi_files())
    return info


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Collect a classical MIDI dataset.")
    parser.add_argument("--export-corpus", action="store_true", help="export pieces from music21's corpus")
    parser.add_argument("--composers", nargs="+", default=list(DEFAULT_COMPOSERS))
    parser.add_argument("--per-composer", type=int, default=15)
    parser.add_argument("--url", help="download a .zip of MIDI files from this URL")
    args = parser.parse_args()

    ensure_dirs()
    if args.export_corpus:
        print(export_music21_corpus(args.composers, args.per_composer, progress=lambda i, n, f: print(f"[{i}/{n}] {f}")))
    if args.url:
        print(download_midi_zip(args.url))
    print(f"MIDI files available: {len(list_midi_files())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
