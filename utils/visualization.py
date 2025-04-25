
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(preds, truth, quantiles, title='Forecast vs Actual', index=0):
    plt.figure(figsize=(10, 4))
    x = np.arange(truth.shape[1])
    plt.plot(x, truth[index].detach().cpu().numpy(), label='Ground Truth', color='black')
    for i, q in enumerate(quantiles):
        plt.plot(x, preds[index, :, i].detach().cpu().numpy(), label=f'Quantile {q}')
    plt.legend()
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
