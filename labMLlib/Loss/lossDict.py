import torch.nn as nn
import inspect
import torch


# loss metrics function


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        numerator = torch.sum(
            (y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true))
        )
        denominator = torch.sqrt(
            (torch.sum((y_pred - torch.mean(y_pred)) ** 2))
            * (torch.sum((y_true - torch.mean(y_true)) ** 2))
        )
        r2_value = (numerator / denominator) ** 2
        return r2_value


class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        numerator = torch.sum((y_true - y_pred) ** 2)
        denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)
        nse_value = 1 - (numerator / denominator)
        return nse_value

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

class lossDict(dict):
    def __init__(self, *arg, **kw):
        super(lossDict, self).__init__(*arg, **kw)
        clsmembers = inspect.getmembers(nn, inspect.isclass)
        self.kw = kw
        for i, cls in clsmembers:
            if "Loss" in i:
                self.update({i: cls})
        for i in [RMSELoss, R2Loss, NSELoss, MAE, MAPE]:
            self.update({i.__name__: i})

    def __getitem__(self, __key):
        available_kw = {}
        for arg in self.kw.keys():
            if arg in inspect.getfullargspec(self.get(__key)).args:
                available_kw[arg] = self.kw[arg]
        return (
            self.get(__key)(**self.kw)
            if isinstance(self.get(__key), type)
            else self.get(__key)
        )


if __name__ == "__main__":
    # demostration of nnDict
    d = lossDict()
    print(d.items())
    for k, v in d.items():
        print(k, type(v))
    loss = d["MSELoss"]
    t1, t2 = torch.rand((1, 64, 64)), torch.ones((1, 64, 64))
    # clsmembers = inspect.getmembers(nn, inspect.isclass)
    # for i,cls in clsmembers:
    #     if "Loss" in i:
    #         print(i,cls)
    print(loss(t1, t2))
