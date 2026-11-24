"""Lock-Free Audio Ring Buffer & File/Stream Input Adapters."""

import time
import numpy as np
from scipy.io import wavfile

class AudioRingBuffer:
    """Fixed-size circular NumPy buffer for lock-free audio streaming."""

    def __init__(self, capacity: int = 44100 * 3):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_ptr = 0

    def write(self, samples: np.ndarray):
        """Writes 1D float32 audio samples into the circular ring buffer."""
        samples = samples.astype(np.float32)
        n = len(samples)
        if n >= self.capacity:
            # If batch exceeds capacity, take the newest portion
            self.buffer[:] = samples[-self.capacity:]
            self.write_ptr = 0
            return

        end_ptr = self.write_ptr + n
        if end_ptr <= self.capacity:
            self.buffer[self.write_ptr:end_ptr] = samples
        else:
            first_part = self.capacity - self.write_ptr
            second_part = n - first_part
            self.buffer[self.write_ptr:] = samples[:first_part]
            self.buffer[:second_part] = samples[first_part:]

        self.write_ptr = (self.write_ptr + n) % self.capacity

    def read_recent(self, n_samples: int) -> np.ndarray:
        """Retrieves the latest n_samples chronologically."""
        if n_samples > self.capacity:
            n_samples = self.capacity

        if self.write_ptr >= n_samples:
            return self.buffer[self.write_ptr - n_samples : self.write_ptr].copy()
        else:
            tail_len = n_samples - self.write_ptr
            return np.concatenate((
                self.buffer[self.capacity - tail_len:],
                self.buffer[:self.write_ptr]
            ))

def load_wav_file(file_path: str) -> tuple[int, np.ndarray]:
    """Loads WAV file and normalizes to mono float32 (-1.0 to 1.0)."""
    sr, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert stereo to mono

    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128.0) / 128.0
    else:
        samples = data.astype(np.float32)

    return sr, samples
