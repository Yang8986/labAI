import torch
import torch.nn as nn
import torch.optim as optim


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


class absMSELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, preds, answer):
        return torch.mean(torch.abs(preds - answer))

class MAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, preds, answer):
        return torch.mean(torch.abs(preds - answer))


class MAPE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, preds, answer):
        return torch.mean(torch.abs((preds - answer)/(answer+1e-8)))*100