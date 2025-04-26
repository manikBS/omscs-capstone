import torch

def forecaster(model_class,
             model_path: str,
             past_window: torch.Tensor,
             forecast_steps: int,
             input_dim: int,
             seq_len: int,
             quantiles=[0.1, 0.5, 0.9],
             device: str = 'cpu',
             **model_kwargs) -> torch.Tensor:
    """
    Load a trained only transformer model from a checkpoint and forecast.
    """
    # Initialize model and load weights
    model = model_class(
        input_dim=input_dim,
        d_model=model_kwargs.get('d_model', 64),
        n_heads=model_kwargs.get('n_heads', 4),
        ff_dim=model_kwargs.get('ff_dim', 128),
        n_layers=model_kwargs.get('n_layers', 3),
        seq_len=seq_len,
        target_len=forecast_steps,
        quantiles=quantiles
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare input
    past_window = past_window.unsqueeze(0).to(device)  # (1, seq_len, input_dim)
    x_dec = torch.zeros((1, forecast_steps, input_dim), device=device)

    with torch.no_grad():
        preds = model(past_window, x_dec)  # (1, forecast_steps, num_quantiles)
        median_forecast = preds[0, :, quantiles.index(0.5)]  # (forecast_steps,)
        return median_forecast.cpu()
