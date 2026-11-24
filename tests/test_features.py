"""Unit tests for acoustic feature extraction."""

import numpy as np
from beat_lens.features import FeatureExtractor

def test_feature_extractor_shape_and_range():
    sr = 44100
    fe = FeatureExtractor(sample_rate=sr, n_fft=2048, n_mfcc=13)

    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    vec = fe.extract(sig)
    # 5 standard features + 13 MFCCs = 18 elements
    assert len(vec) == 18
    assert not np.any(np.isnan(vec))
    assert not np.any(np.isinf(vec))

    # Centroid for 440 Hz pure sine should be around 440 Hz
    centroid = vec[0]
    assert 380 <= centroid <= 500
