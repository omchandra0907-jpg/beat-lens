"""DSP Engine: Hann Windowing, rFFT, and 48-Band Logarithmic Binning."""

import numpy as np

NUM_BANDS = 48
MIN_FREQ = 20.0
MAX_FREQ = 16000.0

class SpectrumAnalyzer:
    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = np.hanning(fft_size)

        # 48 logarithmically spaced frequency boundaries
        self.band_edges = np.geomspace(MIN_FREQ, MAX_FREQ, NUM_BANDS + 1)
        self.fft_freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        # Precompute bin masks for each band
        self.band_masks = []
        for i in range(NUM_BANDS):
            low = self.band_edges[i]
            high = self.band_edges[i + 1]
            mask = (self.fft_freqs >= low) & (self.fft_freqs < high)
            # Ensure at least 1 bin per band
            if not np.any(mask):
                nearest_idx = np.argmin(np.abs(self.fft_freqs - (low + high) / 2))
                mask[nearest_idx] = True
            self.band_masks.append(mask)

        self.smooth_energy = np.zeros(NUM_BANDS, dtype=np.float32)
        self.peak_hold = np.zeros(NUM_BANDS, dtype=np.float32)

    def process(self, audio_chunk: np.ndarray, smoothing: float = 0.65, decay: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
        """Computes 48 logarithmic band magnitudes with exponential smoothing and peak-hold caps."""
        if len(audio_chunk) < self.fft_size:
            padded = np.zeros(self.fft_size, dtype=np.float32)
            padded[-len(audio_chunk):] = audio_chunk
            audio_chunk = padded
        else:
            audio_chunk = audio_chunk[-self.fft_size:]

        # 1. Apply Hann window
        windowed = audio_chunk * self.window

        # 2. Compute Real FFT magnitudes
        rfft = np.fft.rfft(windowed)
        magnitudes = (np.abs(rfft) / self.fft_size) * 2.0

        # 3. Logarithmic band pooling
        raw_bands = np.zeros(NUM_BANDS, dtype=np.float32)
        for i, mask in enumerate(self.band_masks):
            val = np.mean(magnitudes[mask]) if np.any(mask) else 0.0
            # Apply perceptual log boost for high frequencies
            freq_boost = 1.0 + (i / NUM_BANDS) * 2.5
            raw_bands[i] = val * freq_boost

        # Normalize and clip
        scaled = np.clip(raw_bands * 18.0, 0.0, 1.0)

        # 4. Exponential smoothing & Peak-Hold Decay
        self.smooth_energy = smoothing * scaled + (1.0 - smoothing) * self.smooth_energy
        self.peak_hold = np.maximum(self.smooth_energy, self.peak_hold - decay)

        return self.smooth_energy.copy(), self.peak_hold.copy()

    def get_band_for_freq(self, freq: float) -> int:
        """Returns the 0..47 band index where a given frequency resides."""
        for i in range(NUM_BANDS):
            if self.band_edges[i] <= freq < self.band_edges[i + 1]:
                return i
        if freq < self.band_edges[0]:
            return 0
        return NUM_BANDS - 1
