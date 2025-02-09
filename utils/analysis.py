import numpy as np
from scipy import stats, signal


def calculate_signal_stats(data):
    """
    Calculate basic signal statistics.
    """
    stats_dict = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'rms': np.sqrt(np.mean(np.square(data))),
        'kurtosis': stats.kurtosis(data),
        'skewness': stats.skew(data)
    }
    return stats_dict


def detect_anomalies(data, threshold=3):
    """
    Detect anomalies using z-score method.
    """
    z_scores = np.abs(stats.zscore(data))
    anomalies = np.where(z_scores > threshold)[0]
    return anomalies


def analyze_frequency_components(data, sampling_rate=1000):
    """
    Analyze frequency components of the signal.
    """
    freqs, psd = signal.welch(data, fs=sampling_rate)
    dominant_freq = freqs[np.argmax(psd)]


    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    band_powers = {}
    for band, (low, high) in freq_bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_powers[band] = np.sum(psd[mask])

    return {
        'dominant_frequency': dominant_freq,
        'band_powers': band_powers
    }


def detect_bursts(data, threshold=2):
    """
    Detect burst activities in neural signal.
    """
    envelope = np.abs(signal.hilbert(data))
    threshold_value = np.mean(envelope) + threshold * np.std(envelope)
    bursts = np.where(envelope > threshold_value)[0]


    burst_events = []
    if len(bursts) > 0:
        burst_start = bursts[0]
        for i in range(1, len(bursts)):
            if bursts[i] - bursts[i - 1] > 1:
                burst_events.append({
                    'start': burst_start,
                    'end': bursts[i - 1],
                    'duration': bursts[i - 1] - burst_start
                })
                burst_start = bursts[i]
        # Add last burst
        burst_events.append({
            'start': burst_start,
            'end': bursts[-1],
            'duration': bursts[-1] - burst_start
        })

    return burst_events


def calculate_coherence(signal1, signal2, sampling_rate=1000):
    """
    Calculate coherence between two signals.
    """
    freqs, coherence = signal.coherence(signal1, signal2, fs=sampling_rate)
    mean_coherence = np.mean(coherence)
    max_coherence = np.max(coherence)
    coherence_freq = freqs[np.argmax(coherence)]

    return {
        'frequencies': freqs,
        'coherence': coherence,
        'mean_coherence': mean_coherence,
        'max_coherence': max_coherence,
        'max_coherence_freq': coherence_freq
    }