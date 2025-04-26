
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def generate_data_from_df(df, feature_cols, input_len, pred_len, normalize=True):

    data = df[feature_cols].copy().to_numpy()

    if normalize:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data) - input_len - pred_len):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + pred_len, feature_cols.index("close")])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

    return X_tensor, y_tensor, scaler


def invert_scale(pred_tensor, scaler, feature_index, num_features):
    """
    Inverts MinMax scaling for each quantile forecast separately.
    """
    pred_np = pred_tensor.detach().cpu().numpy()
    B, T, Q = pred_np.shape
    unscaled_preds = np.zeros_like(pred_np)

    for q in range(Q):
        flat_preds = pred_np[:, :, q].reshape(-1, 1)  # (B*T, 1)
        dummy = np.zeros((flat_preds.shape[0], num_features))
        dummy[:, feature_index] = flat_preds[:, 0]

        unscaled = scaler.inverse_transform(dummy)[:, feature_index]
        unscaled_preds[:, :, q] = unscaled.reshape(B, T)

    return unscaled_preds


def invert_truth_scale(truth_tensor, scaler, feature_index, num_features):
    """
    Inverts the MinMax scaling of ground truth values.
    """
    truth_np = truth_tensor.detach().cpu().numpy()
    B, T = truth_np.shape

    flat_truth = truth_np.reshape(-1, 1)  # (B*T, 1)
    dummy_input = np.zeros((flat_truth.shape[0], num_features))
    dummy_input[:, feature_index] = flat_truth[:, 0]

    unscaled = scaler.inverse_transform(dummy_input)[:, feature_index]
    unscaled_truth = unscaled.reshape(B, T)

    return unscaled_truth

def plot_unscaled_predictions(unscaled_preds, unscaled_truth, quantiles, index=0, title="Forecast vs Ground Truth (Unscaled)"):
    x = np.arange(unscaled_preds.shape[1])
    plt.figure(figsize=(10, 4))

    plt.plot(x, unscaled_truth[index], label="Ground Truth", color="black", linewidth=2)

    for i, q in enumerate(quantiles):
        plt.plot(x, unscaled_preds[index, :, i], label=f"Quantile {q}", linestyle="--")

    plt.title(title)
    plt.xlabel("Forecast Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
