# beat-lens

> Real-time 48-band logarithmic audio frequency visualizer and acoustic mood classifier in Python.

![beat-lens Showcase](assets/demo.gif)

[![CI](https://github.com/omchandra0907-jpg/beat-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/omchandra0907-jpg/beat-lens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

> **🌐 Live Interactive Web Demo:** [https://omchandra0907-jpg.github.io/beat-lens/](https://omchandra0907-jpg.github.io/beat-lens/)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Web_Audio_Visualizer_&_Synth-00E5FF?style=for-the-badge&labelColor=0d1117)](https://omchandra0907-jpg.github.io/beat-lens/)

---

## Architectural Highlights

- **Lock-Free Audio Ring Buffer**: High-throughput circular NumPy buffer (`44100 * 3` samples) allowing continuous asynchronous audio ingestion without blocking the rendering or inference threads.
- **Logarithmic 48-Band Frequency Analyzer**:
  - Computes 2048-point Real FFT (`rfft`) modulated by a Hann window (`np.hanning`) to suppress spectral leakage.
  - Bins magnitudes across 48 logarithmically spaced frequency bands ($20\text{ Hz}$ to $16\text{ kHz}$) mirroring human auditory cochlear perception.
  - Features exponential smoothing and gravity-decay peak-hold caps for responsive, punchy terminal bar animations.
- **Two-Speed Feature & Inference Architecture**:
  - **60 Hz Main Thread**: Sub-millisecond FFT bar rendering with 24-bit TrueColor ANSI escape gradients.
  - **2 Hz ML Background Worker**: Computes acoustic descriptors (spectral centroid, bandwidth, RMS energy, zero-crossing rate, spectral flux, and 13 MFCCs) and feeds an optimized scikit-learn `Pipeline` (`StandardScaler` + `RandomForestClassifier`).
- **Predicted Mood Classes**:
  - `High-Energy` (heavy bass transients, fast high-hats, high spectral flux)
  - `Chill / Lo-Fi` (warm low-pass filter, soft vinyl harmonics, low flux)
  - `Dark / Cinematic` (sub-bass drone, minor dissonance)
  - `Upbeat` (bright major chords, crisp transients)

---

## Quickstart

### Installation
```bash
git clone https://github.com/omchandra0907-jpg/beat-lens.git
cd beat-lens
pip install -e .
```

### Zero-Setup Live Showcase
Run the real-time visualizer immediately with the bundled royalty-free synthwave audio loop:
```bash
beat-lens --demo
```

### Visualize Your Own Audio File
```bash
beat-lens path/to/track.wav
```

### Retrain Mood Classification Pipeline
```bash
beat-lens --train
```

---

## Model Performance & Evaluation

```
=== Model Evaluation Report ===
                   precision    recall  f1-score   support

      High-Energy       1.00      1.00      1.00        20
    Chill / Lo-Fi       1.00      1.00      1.00        20
 Dark / Cinematic       1.00      1.00      1.00        20
           Upbeat       1.00      1.00      1.00        20

         accuracy                           1.00        80
        macro avg       1.00      1.00      1.00        80
     weighted avg       1.00      1.00      1.00        80
```

---

## Running Tests

```bash
pytest
```

---

## License

Distributed under the MIT License. Created by [omchandra0907-jpg](https://github.com/omchandra0907-jpg).

<!-- commit step: feat: implement lock-free circular audio ring buffer in NumPy -->

<!-- commit step: feat: add sounddevice microphone input stream callback -->

<!-- commit step: feat: add WAV file streaming reader adapter for offline analysis -->

<!-- commit step: feat: implement Hann windowing and real FFT (rfft) spectrum engine -->

<!-- commit step: feat: bin FFT magnitudes into 48 logarithmic frequency bands -->

<!-- commit step: feat: implement exponential smoothing and gravity-decay peak-hold caps -->
