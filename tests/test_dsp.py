"""Unit tests for DSP frequency analysis, peak identification, and ring buffer."""

import numpy as np
import pytest
from beat_lens.dsp import SpectrumAnalyzer, NUM_BANDS
from beat_lens.stream import AudioRingBuffer

def test_ring_buffer_fifo_and_wrap():
    buf = AudioRingBuffer(capacity=100)
    data1 = np.ones(60, dtype=np.float32)
    buf.write(data1)

    rec = buf.read_recent(30)
    assert len(rec) == 30
    assert np.all(rec == 1.0)

    # Overflow wrap
    data2 = np.full(70, 2.0, dtype=np.float32)
    buf.write(data2)

    rec_all = buf.read_recent(100)
    # The oldest 30 should be 1.0, and the latest 70 should be 2.0
    assert np.all(rec_all[:30] == 1.0)
    assert np.all(rec_all[30:] == 2.0)

@pytest.mark.parametrize("target_freq", [100.0, 440.0, 8000.0])
def test_pure_sine_wave_peak_detection(target_freq):
    sr = 44100
    analyzer = SpectrumAnalyzer(sample_rate=sr, fft_size=2048)

    # Generate 0.5s pure sine wave at target frequency (normalized amplitude 0.25)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sine = (0.25 * np.sin(2 * np.pi * target_freq * t)).astype(np.float32)

    energies, peaks = analyzer.process(sine, smoothing=1.0)

    peak_band = int(np.argmax(energies))
    expected_band = analyzer.get_band_for_freq(target_freq)

    # Peak band should match the expected logarithmic bin (within 1 bin edge tolerance)
    assert abs(peak_band - expected_band) <= 1, (
        f"For {target_freq} Hz: expected band ~{expected_band}, got peak in band {peak_band}"
    )
