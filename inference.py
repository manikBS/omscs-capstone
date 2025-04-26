
import torch
import matplotlib.pyplot as plt
from model.patchtst import PatchTST_SOTA
from utils.visualization import plot_predictions

def load_model(checkpoint_path, input_len, pred_len, num_features):
    model = PatchTST_SOTA(input_len=input_len, pred_len=pred_len, num_features=num_features)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_inference(model, test_loader, quantiles):
    model.eval()
    all_preds = []
    all_truth = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            all_preds.append(preds)
            all_truth.append(y_batch)
    preds_tensor = torch.cat(all_preds, dim=0)
    truth_tensor = torch.cat(all_truth, dim=0)
    return preds_tensor, truth_tensor
