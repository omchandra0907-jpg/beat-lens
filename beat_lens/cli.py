"""CLI Entry Point for beat-lens."""

import argparse
import os
import sys
import threading
import time
import joblib
import numpy as np

from .dsp import SpectrumAnalyzer
from .features import FeatureExtractor
from .stream import AudioRingBuffer, load_wav_file
from .train import train_mood_model, generate_synthetic_audio, MOOD_LABELS
from .visualizer import TerminalVisualizer

def run_visualizer_loop(audio_data: np.ndarray, sample_rate: int = 44100, model_path: str = "models/mood_clf.joblib"):
    ring_buf = AudioRingBuffer(capacity=sample_rate * 4)
    dsp = SpectrumAnalyzer(sample_rate=sample_rate, fft_size=2048)
    fe = FeatureExtractor(sample_rate=sample_rate)
    viz = TerminalVisualizer(num_bands=48)

    # Load pre-trained mood classifier if present
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            pass

    stop_event = threading.Event()

    # Background audio streamer thread
    def audio_feeder():
        chunk_size = 512
        interval = chunk_size / sample_rate
        total_len = len(audio_data)
        idx = 0

        while not stop_event.is_set():
            end = idx + chunk_size
            if end <= total_len:
                chunk = audio_data[idx:end]
                idx = end
            else:
                chunk = np.concatenate((audio_data[idx:], audio_data[: end % total_len]))
                idx = end % total_len

            ring_buf.write(chunk)
            time.sleep(interval)

    # Background ML mood inference worker (runs every 500 ms)
    def ml_worker():
        while not stop_event.is_set():
            time.sleep(0.5)
            recent_samples = ring_buf.read_recent(int(sample_rate * 1.5))
            if len(recent_samples) > 2048 and model:
                try:
                    features = fe.extract(recent_samples).reshape(1, -1)
                    probs = model.predict_proba(features)[0]
                    top_idx = int(np.argmax(probs))
                    viz.update_mood(list(probs), MOOD_LABELS[top_idx])
                except Exception:
                    pass

    t_feed = threading.Thread(target=audio_feeder, daemon=True)
    t_ml = threading.Thread(target=ml_worker, daemon=True)

    t_feed.start()
    t_ml.start()

    # Enter alternate buffer and hide cursor
    sys.stdout.write("\033[?1049h\033[?25l\033[2J")
    sys.stdout.flush()

    try:
        frame_interval = 1.0 / 60.0
        while True:
            t0 = time.time()
            chunk = ring_buf.read_recent(2048)
            energies, peaks = dsp.process(chunk)
            viz.render_frame(energies, peaks)

            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # Restore terminal
        sys.stdout.write("\033[?25h\033[?1049l\033[0m\n")
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="beat-lens: Real-Time Audio Frequency Visualizer & Mood Classifier")
    parser.add_argument("audio_file", nargs="?", help="Path to input .wav audio file")
    parser.add_argument("--demo", action="store_true", help="Launch live demo showcase with synthesized audio loop")
    parser.add_argument("--train", action="store_true", help="Retrain reproducible mood classification pipeline")
    parser.add_argument("--model", default="models/mood_clf.joblib", help="Model artifact path")

    args = parser.parse_args()

    if args.train:
        train_mood_model(args.model)
        return

    if args.demo or (not args.audio_file and len(sys.argv) == 1):
        sample_path = "samples/synthwave_loop.wav"
        if os.path.exists(sample_path):
            sr, audio = load_wav_file(sample_path)
        else:
            sr = 44100
            # Generate looping synthwave sample
            audio = generate_synthetic_audio(mood_idx=0, sr=sr, duration=8.0)
        run_visualizer_loop(audio, sample_rate=sr, model_path=args.model)
    elif args.audio_file:
        if not os.path.exists(args.audio_file):
            print(f"[-] Error: File not found: {args.audio_file}", file=sys.stderr)
            sys.exit(1)
        sr, audio = load_wav_file(args.audio_file)
        run_visualizer_loop(audio, sample_rate=sr, model_path=args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
