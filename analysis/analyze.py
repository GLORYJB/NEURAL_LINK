# analysis/analyze.py

import numpy as np

def extract_features(data):
    """
    Extract features from neural data (e.g., mean, variance, frequency components).

    Parameters:
        data (pd.DataFrame): Preprocessed neural data.

    Returns:
        dict: Extracted features.
    """
    features = {
        'mean': np.mean(data, axis=0),
        'variance': np.var(data, axis=0),
        'max': np.max(data, axis=0),
        'min': np.min(data, axis=0)
    }
    return features

def detect_anomalies(data, threshold=3):
    """
    Detect anomalies in neural data using a simple thresholding method.

    Parameters:
        data (pd.DataFrame): Preprocessed neural data.
        threshold (float): Threshold for anomaly detection.

    Returns:
        pd.DataFrame: Data with anomalies flagged.
    """
    anomalies = data[np.abs(data) > threshold]
    return anomalies