"""Streamlit web interface for the AI Music Generator.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from config import (
    BATCH_SIZE,
    DEFAULT_COMPOSERS,
    DEFAULT_EVENTS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPO,
    DEFAULT_TOP_K,
    DROPOUT,
    EPOCHS,
    INSTRUMENTS,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_FILES,
    MIDI_DIR,
    MODEL_FILE,
    SEQUENCE_LENGTH,
    ensure_dirs,
)
from dataset import (
    download_midi_zip,
    export_music21_corpus,
    list_midi_files,
    save_uploaded_midi,
)
from preprocess import DatasetError, build_dataset, dataset_ready, load_stats
from visualize import (
    plot_duration_distribution,
    plot_piano_roll,
    plot_pitch_classes,
    plot_pitch_range,
    plot_training_history,
    tokens_duration_counts,
    tokens_pitch_histogram,
)

st.set_page_config(page_title="AI Music Generator", page_icon="🎼", layout="wide")
ensure_dirs()

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 3.5rem; max-width: 1150px;}
      .hero {background: linear-gradient(120deg,#5B3FE0 0%,#7C5CFF 55%,#0E8F80 130%);
             color:#fff; padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;}
      .hero h1 {margin:0; font-size: 2rem; color:#fff;}
      .hero p {margin:.4rem 0 0 0; opacity:.92; font-size:1.02rem;}
      div[data-testid="stMetric"] {background:#F6F5FF; border:1px solid #E6E2FF;
             padding:.8rem 1rem; border-radius:14px;}
      .stButton>button[kind="primary"] {border-radius: 12px; font-weight: 600;}
      .step {color:#6B7280; font-size:.9rem; margin-bottom:.4rem;}
    </style>
    <div class="hero">
      <h1>🎼 AI Music Generator</h1>
      <p>An LSTM neural network trained from scratch on classical MIDI -
         build a dataset, train, then compose new music you can play and download.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading trained model...")
def get_artifacts(model_mtime: float, data_mtime: float):
    """Cached model + dataset. The mtimes invalidate the cache after retraining."""
    from generate import load_generator_artifacts

    return load_generator_artifacts()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _progress_callback(bar, label):
    def update(i: int, total: int, name: str):
        bar.progress(i / max(total, 1), text=f"{label} {i}/{total}: {name}")

    return update


tab_data, tab_train, tab_gen, tab_about = st.tabs(
    ["1 · Dataset", "2 · Train", "3 · Generate", "About"]
)

# ================================================================== DATASET
with tab_data:
    n_files = len(list_midi_files())
    stats = load_stats() if dataset_ready() else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MIDI files in folder", n_files)
    c2.metric("Files used", stats["files_used"] if stats else "-")
    c3.metric("Vocabulary (tokens)", stats["vocab_size"] if stats else "-")
    c4.metric("Training sequences", f"{stats['n_sequences']:,}" if stats else "-")

    if n_files == 0:
        st.warning(
            f"No MIDI files found in `{MIDI_DIR}`. Export classical pieces below, upload your own "
            "files, or copy `.mid` files into that folder."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Collect classical MIDI")
        st.caption("Exports pieces from the corpus bundled with music21 (works offline).")
        composers = st.multiselect(
            "Composers",
            ["bach", "mozart", "beethoven", "haydn", "handel", "schumann_robert", "schumann_clara",
             "cpebach", "weber", "verdi", "monteverdi", "joplin"],
            default=list(DEFAULT_COMPOSERS),
        )
        per_composer = st.slider("Pieces per composer", 3, 60, 15)
        if st.button("Export classical pieces", type="primary", disabled=not composers):
            bar = st.progress(0.0)
            try:
                result = export_music21_corpus(composers, per_composer, progress=_progress_callback(bar, "Exporting"))
                bar.empty()
                st.success(
                    f"Wrote {result['written']} new files ({result['already_present']} already present, "
                    f"{result['failed']} failed)."
                )
                if result["errors"]:
                    with st.expander("Problems"):
                        st.write(result["errors"])
            except Exception as exc:
                bar.empty()
                st.error(f"Export failed: {exc}")

    with right:
        st.subheader("Add your own files")
        uploads = st.file_uploader("Upload .mid / .midi files", type=["mid", "midi"], accept_multiple_files=True)
        if uploads and st.button("Save uploaded files"):
            for up in uploads:
                ok, message = save_uploaded_midi(up.name, up.getvalue())
                (st.success if ok else st.error)(message)
        url = st.text_input("...or a URL to a .zip of MIDI files", placeholder="https://example.com/classical.zip")
        if url and st.button("Download zip"):
            bar = st.progress(0.0)
            try:
                result = download_midi_zip(url.strip(), progress=_progress_callback(bar, "Extracting"))
                bar.empty()
                st.success(f"Saved {result['saved']} MIDI files ({result['invalid']} invalid skipped).")
            except Exception as exc:
                bar.empty()
                st.error(str(exc))

    st.divider()
    st.subheader("Preprocess with music21")
    st.markdown(
        '<div class="step">Extracts notes and chords, transposes to C major / A minor, '
        "builds the vocabulary and sliding-window sequences.</div>",
        unsafe_allow_html=True,
    )
    p1, p2 = st.columns(2)
    seq_len = p1.slider("Sequence length (tokens of context)", 32, 192, SEQUENCE_LENGTH, step=16)
    max_files = p2.slider("Max files to parse", 10, 500, MAX_FILES, step=10)
    if st.button("Preprocess dataset", type="primary"):
        bar = st.progress(0.0)
        try:
            build_dataset(sequence_length=seq_len, max_files=max_files, progress=_progress_callback(bar, "Parsing"))
            bar.empty()
            st.success("Dataset ready. Go to the Train tab.")
            st.rerun()
        except DatasetError as exc:
            bar.empty()
            st.error(str(exc))
        except Exception as exc:  # last-resort guard so the app never crashes
            bar.empty()
            st.error(f"Unexpected preprocessing error: {exc}")

    if stats:
        st.subheader("Dataset overview")
        g1, g2 = st.columns(2)
        g1.pyplot(plot_pitch_classes(stats["pitch_histogram"], "Pitch classes in the dataset"))
        g2.pyplot(plot_duration_distribution(stats["duration_counts"], "Durations in the dataset"))
        st.pyplot(plot_pitch_range(stats["pitch_histogram"]))
        plt.close("all")
        if stats["skipped"]:
            with st.expander(f"{stats['files_skipped']} files were skipped"):
                st.dataframe(stats["skipped"], width="stretch")
        with st.expander("Most common tokens"):
            st.dataframe(stats["top_tokens"], width="stretch")

# ==================================================================== TRAIN
with tab_train:
    if not dataset_ready():
        st.info("Preprocess a dataset first (tab 1).")
    else:
        st.subheader("Train the LSTM")
        t1, t2, t3, t4 = st.columns(4)
        epochs = t1.slider("Max epochs", 1, 100, EPOCHS)
        units = t2.select_slider("LSTM units", [64, 128, 256, 384, 512], value=LSTM_UNITS)
        batch = t3.select_slider("Batch size", [32, 64, 128, 256], value=BATCH_SIZE)
        lr = t4.select_slider("Learning rate", [3e-4, 5e-4, 1e-3, 2e-3], value=LEARNING_RATE)
        st.caption("Early stopping ends training once validation loss stops improving; the best weights are kept.")

        if st.button("Start training", type="primary"):
            from train import train_model  # imports TensorFlow lazily

            history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
            bar = st.progress(0.0, text="Starting...")
            live_metrics = st.empty()
            live_chart = st.empty()

            def on_epoch(epoch, total, logs):
                for key in history:
                    history[key].append(logs.get(key, float("nan")))
                bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
                live_metrics.markdown(
                    f"**loss** {logs.get('loss', 0):.3f} · **acc** {logs.get('accuracy', 0):.3f} · "
                    f"**val loss** {logs.get('val_loss', 0):.3f} · **val acc** {logs.get('val_accuracy', 0):.3f}"
                )
                fig = plot_training_history(history)
                live_chart.pyplot(fig)
                plt.close(fig)

            try:
                with st.spinner("Training... this can take a few minutes on CPU."):
                    train_model(epochs, batch, units, dropout=DROPOUT, learning_rate=lr, on_epoch_end=on_epoch)
                bar.progress(1.0, text="Training complete")
                st.success("Model trained and saved. Open the Generate tab.")
            except DatasetError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        from train import load_history, load_training_info

        info, hist = load_training_info(), load_history()
        if info and hist:
            st.divider()
            st.subheader("Latest training run")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best val. accuracy", f"{info['best_val_accuracy'] * 100:.1f}%")
            m2.metric("Best val. loss", f"{info['best_val_loss']:.3f}")
            m3.metric("Epochs run", f"{info['epochs_run']} (best #{info['best_epoch']})")
            m4.metric("Parameters", f"{info['parameters']:,}")
            st.pyplot(plot_training_history(hist))
            plt.close("all")
            st.caption(
                f"Perplexity {info['best_val_perplexity']:.1f} over a vocabulary of {info['vocab_size']} tokens "
                f"· {info['train_sequences']:,} train / {info['val_sequences']:,} validation sequences "
                f"· {info['seconds']}s · TensorFlow {info['tensorflow']} · {info['trained_at']}"
            )
            with st.expander("Model architecture"):
                st.code(info["model_summary"])

# ================================================================= GENERATE
with tab_gen:
    if not MODEL_FILE.exists() or not dataset_ready():
        st.info("Train a model first (tabs 1 and 2) to start composing.")
    else:
        st.subheader("Compose new music")
        g1, g2, g3 = st.columns(3)
        events = g1.slider("Length (events)", 32, 400, DEFAULT_EVENTS, step=16)
        temperature = g2.slider("Creativity (temperature)", 0.3, 1.6, DEFAULT_TEMPERATURE, step=0.05,
                                help="Low = safe and repetitive, high = adventurous and risky.")
        top_k = g3.slider("Top-k sampling", 1, 40, DEFAULT_TOP_K)
        g4, g5, g6 = st.columns(3)
        tempo = g4.slider("Tempo (BPM)", 50, 180, DEFAULT_TEMPO)
        instrument = g5.selectbox("MIDI instrument", list(INSTRUMENTS))
        fixed_seed = g6.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)

        if st.button("Generate music", type="primary"):
            try:
                from generate import ModelNotFoundError, generate_music

                art = get_artifacts(_mtime(MODEL_FILE), _mtime(Path(MODEL_FILE).parent.parent / "data" / "processed" / "dataset.npz"))
                with st.spinner("Composing..."):
                    st.session_state["gen"] = generate_music(
                        art, events, temperature, top_k, tempo, INSTRUMENTS[instrument],
                        rng_seed=int(fixed_seed) or None,
                    )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.session_state.pop("gen", None)

        result = st.session_state.get("gen")
        if result:
            st.divider()
            s = result["stats"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Duration", f"{s['duration_seconds']} s")
            m2.metric("Notes", s["notes"])
            m3.metric("Unique pitches", s["unique_pitches"])
            m4.metric("Pitch range", s["pitch_range"])
            m5.metric("Novelty", f"{s['novelty'] * 100:.0f}%", help="Share of short phrases (16 tokens, about 4 notes/chords) that never occur in the training data.")

            wav_bytes = Path(result["wav_path"]).read_bytes()
            midi_bytes = Path(result["midi_path"]).read_bytes()
            st.audio(wav_bytes, format="audio/wav")
            st.caption("The player uses a simple built-in synthesizer. The MIDI file sounds richer in any MIDI player.")
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button("⬇ Download MIDI", midi_bytes, file_name=Path(result["midi_path"]).name,
                               mime="audio/midi", type="primary")
            d2.download_button("⬇ Download WAV", wav_bytes, file_name=Path(result["wav_path"]).name, mime="audio/wav")

            st.pyplot(plot_piano_roll(result["tokens"], result["tempo"]))
            c1, c2 = st.columns(2)
            c1.pyplot(plot_pitch_classes(tokens_pitch_histogram(result["tokens"]), "Pitch classes (generated)"))
            c2.pyplot(plot_duration_distribution(tokens_duration_counts(result["tokens"]), "Durations (generated)"))
            plt.close("all")

# ==================================================================== ABOUT
with tab_about:
    st.markdown(
        """
### How it works
1. **Collect** classical MIDI files (your own, a zip URL, or music21's bundled corpus).
2. **Extract** notes and chords with **music21**, transpose to C major / A minor, quantise durations.
3. **Tokenise**: every chord/note event (e.g. `60.64.67|1`) becomes atomic tokens `n60 n64 n67 d1`
   (notes, then a duration), each mapped to an integer id. The vocabulary stays around 100 tokens.
4. **Train** a two-layer **LSTM** (TensorFlow/Keras) to predict the next token from the previous
   events. It is trained from scratch - no pretrained music model is used.
5. **Generate** by sampling the network's predictions one event at a time (temperature + top-k).
6. **Export** with **PrettyMIDI** to a playable `.mid` (plus a `.wav` preview).

### Tips
- More and more varied pieces give better music. A few hundred pieces is a good start.
- Lower the temperature for tidy, repetitive phrases; raise it for surprising ones.
- **Novelty** shows how much of the output is not copied from the training data.
        """
    )
