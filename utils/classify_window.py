import torch
import numpy as np

def classify_window(window, model_type='1d', model_path=None, PATTERN_LABELS=[]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    window = np.asarray(window, dtype=np.float32)

    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-8
    window = (window - mean) / std

    if model_type == '1d':
        from model.CNN.CNN1D import CNN1D_MultiChannel
        model = CNN1D_MultiChannel(in_channels=window.shape[1], num_classes=len(PATTERN_LABELS))
        tensor_input = torch.tensor(window.T).unsqueeze(0).to(device)

    elif model_type == '2d':
        from model.CNN.CNN2D import CNN2D_MultiChannel
        model = CNN2D_MultiChannel(in_channels=window.shape[1], num_classes=len(PATTERN_LABELS))
        reshaped = window.T.reshape(window.shape[1], 2, -1)
        tensor_input = torch.tensor(reshaped).unsqueeze(0).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(tensor_input)
        predicted_idx = torch.argmax(output, dim=1).item()

    return PATTERN_LABELS[predicted_idx]