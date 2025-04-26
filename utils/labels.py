from scipy.signal import find_peaks
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas_ta as ta

def detect_head_and_shoulders(close_prices, tolerance=0.03):
    peaks, _ = find_peaks(close_prices)
    if len(peaks) < 3:
        return False

    # Look for three peaks (left, head, right)
    for i in range(len(peaks) - 2):
        left, head, right = peaks[i], peaks[i + 1], peaks[i + 2]
        if head - left > 2 and right - head > 2:  # spacing
            l, h, r = close_prices[left], close_prices[head], close_prices[right]
            if h > l and h > r:
                if abs(l - r) / h < tolerance:
                    return True
    return False


def detect_double_bottom(close_prices, PEAK_PROMINENCE=0.01):
    troughs, _ = find_peaks(-close_prices)
    if len(troughs) < 2:
        return False

    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i + 1]
        if abs(t1 - t2) < 3:
            continue  # too close

        v1, v2 = close_prices[t1], close_prices[t2]
        mid = close_prices[t1:t2].max()

        if abs(v1 - v2) / mid < PEAK_PROMINENCE and mid > v1 and mid > v2:
            return True
    return False


def detect_ascending_triangle(close_prices, PEAK_PROMINENCE=0.01):
    peaks, _ = find_peaks(close_prices)
    troughs, _ = find_peaks(-close_prices)
    if len(peaks) < 2 or len(troughs) < 2:
        return False

    # Check if resistance line is flat and lows are rising
    top_diff = max(close_prices[peaks]) - min(close_prices[peaks])
    low_diffs = np.diff(close_prices[troughs])
    if top_diff / close_prices[peaks[-1]] < PEAK_PROMINENCE and np.all(low_diffs > 0):
        return True
    return False


def label_patterns(df, window_size=10, target_attribute='close'):
    labels = []
    windows = []
    for i in range(len(df) - window_size):

        window = df.iloc[i:i + window_size]
        values = window[target_attribute].values

        if detect_head_and_shoulders(values):
            label = 'head_and_shoulders'
        elif detect_double_bottom(values):
            label = 'double_bottom'
        elif detect_ascending_triangle(values):
            label = 'ascending_triangle'
        else:
            label = 'none'
        windows.append({"window": window, "label": label})
        # labels.append(label)

    # Pad end of DataFrame with None
    # labels += ['none'] * (window_size)
    # df['Pattern'] = labels
    # print(df)
    return windows

def create_windows(df, window_size=10, target_cols=['close','ema','rsi','zscore']):
    """
    Convert a DataFrame with shape (N, 4) into a tensor of shape (num_windows, 30, 4)
    """
    X = []
    targets = []

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['label'])

    for i in range(len(df)):
        X.append(df['window'].iloc[i][target_cols])
        targets.append(df['Label'].iloc[i])


    X = np.stack(X)  # (num_windows, 30, 4)
    tensors = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    return tensors, targets
