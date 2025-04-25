
import torch

def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(2))
    return torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))

def mean_absolute_error(preds, target):
    return torch.mean(torch.abs(preds - target))

def root_mean_squared_error(preds, target):
    return torch.sqrt(torch.mean((preds - target) ** 2))
