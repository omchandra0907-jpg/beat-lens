"""Acoustic & Rhythmic Feature Vector Extraction Engine."""

import numpy as np
from scipy.fftpack import dct

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def compute_mel_filterbank(sr: int = 44100, n_fft: int = 2048, n_mels: int = 26) -> np.ndarray:
    """Constructs triangular Mel filterbank matrix."""
    f_min, f_max = 20.0, sr / 2.0
    mel_min = hz_to_mel(np.array([f_min]))[0]
    mel_max = hz_to_mel(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank

class FeatureExtractor:
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, n_mfcc: int = 13):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.mel_fb = compute_mel_filterbank(sample_rate, n_fft, n_mels=26)
        self.prev_spectrum: np.ndarray | None = None

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extracts 18-element feature vector:
        [centroid, bandwidth, rms, zcr, flux, mfcc_0..mfcc_12]
        """
        if len(audio) < self.n_fft:
            padded = np.zeros(self.n_fft, dtype=np.float32)
            padded[-len(audio):] = audio
            audio = padded
        else:
            audio = audio[-self.n_fft:]

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))

        # 2. Zero Crossing Rate (ZCR)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio)))))

        # Spectrum
        windowed = audio * np.hanning(self.n_fft)
        mag = np.abs(np.fft.rfft(windowed)) + 1e-12
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)

        # 3. Spectral Centroid
        centroid = float(np.sum(freqs * mag) / np.sum(mag))

        # 4. Spectral Bandwidth
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))

        # 5. Spectral Flux
        if self.prev_spectrum is not None:
            flux = float(np.sum(np.maximum(0, mag - self.prev_spectrum)))
        else:
            flux = 0.0
        self.prev_spectrum = mag.copy()

        # 6. MFCCs (13 coefficients)
        mel_energy = np.dot(self.mel_fb, mag)
        log_mel = np.log(mel_energy + 1e-12)
        mfccs = dct(log_mel, type=2, norm='ortho')[:self.n_mfcc]

        features = [centroid, bandwidth, rms, zcr, flux] + list(mfccs)
        return np.array(features, dtype=np.float32)
