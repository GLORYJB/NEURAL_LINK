import pandas as pd
import numpy as np
from scipy import signal


def load_data(file_path):
    """
    Load neural data from CSV file.
    If the file does not exist, create simulated data for testing.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:

        print(f"File {file_path} not found. Generating simulated data.")
        return create_simulated_data()


def create_simulated_data(n_channels=5, n_samples=1000):
    """
    Create simulated neural data for testing purposes.
    Generates random signals with noise, trends, and oscillations.
    """
    np.random.seed(42)
    data = {}


    t = np.linspace(0, 10, n_samples)

    for i in range(n_channels):

        base_signal = (np.sin(2 * np.pi * 0.5 * t) +
                       0.5 * np.sin(2 * np.pi * 1.5 * t))


        noise = np.random.normal(0, 0.2, n_samples)


        trend = np.random.uniform(-0.1, 0.1) * t


        signal = base_signal + noise + trend


        data[f'channel_{i + 1}'] = signal

    return pd.DataFrame(data)


def preprocess_data(data, smoothing_window=5):
    """
    Preprocess neural data by interpolating, smoothing, and normalizing the data.
    """
    processed_data = pd.DataFrame()

    for column in data.columns:
        series = data[column]


        series = series.interpolate(method='linear')


        series = series.rolling(window=smoothing_window, center=True).mean()


        series = (series - series.mean()) / series.std()

        processed_data[column] = series


    processed_data = processed_data.bfill().ffill()

    return processed_data


def detect_patterns(data, threshold=3):
    """
    Detect patterns and anomalies in the neural data.
    Detects trends, oscillations, anomalies, and correlations.
    """
    patterns = {
        'trends': [],
        'oscillations': [],
        'anomalies': [],
        'correlations': []
    }

    for column in data.columns:
        series = data[column]

        # 1. Detect trends
        trend = np.polyfit(range(len(series)), series, 1)[0]
        if abs(trend) > 0.1:
            patterns['trends'].append({
                'channel': column,
                'direction': 'upward' if trend > 0 else 'downward',
                'magnitude': abs(trend)
            })


        freqs, psd = signal.welch(series)
        peak_freq = freqs[np.argmax(psd)]
        if peak_freq > 0.1:
            patterns['oscillations'].append({
                'channel': column,
                'frequency': peak_freq
            })


        zscore = np.abs((series - series.mean()) / series.std())
        anomalies = np.where(zscore > threshold)[0]
        if len(anomalies) > 0:
            patterns['anomalies'].append({
                'channel': column,
                'indices': anomalies,
                'values': series[anomalies]
            })


    corr_matrix = data.corr()
    high_corr = np.where(np.abs(corr_matrix) > 0.7)
    for i, j in zip(*high_corr):
        if i < j:  # Avoid duplicate pairs
            patterns['correlations'].append({
                'channels': (data.columns[i], data.columns[j]),
                'correlation': corr_matrix.iloc[i, j]
            })

    return patterns
