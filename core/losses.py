import torch

from core.utils import BetaInc


def rand_loss(y_target, y_pred, theta, n=100):

    w_theta = torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1)

    return torch.stack([BetaInc.apply(torch.tensor(n // 2 + 1), torch.tensor(n // 2), w) for w in w_theta]).mean()

def moment_loss(y_target, y_pred, theta, order=1):

    assert order in [1, 2], "only first and second order supported atm"

    w_theta = torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1)
    
    return (w_theta ** order).mean()

def loss_binary(y_target, y_pred, alpha):

    correct = torch.where(y_target == y_pred, alpha, torch.zeros(1)).sum(1)
    wrong = torch.where(y_target != y_pred, alpha, torch.zeros(1)).sum(1)

    s = [BetaInc.apply(c, w, torch.tensor(0.5)) for c, w in zip(correct, wrong)]

    return sum(s) / len(y_target)

def margin_loss_binary(y_target, y_pred, alpha, logit_gamma):

    correct = torch.where(y_target == y_pred, alpha, torch.zeros(1)).sum(1)
    wrong = torch.where(y_target != y_pred, alpha, torch.zeros(1)).sum(1)

    s = [BetaInc.apply(c, w, torch.tensor(0.5) + torch.sigmoid(logit_gamma)/2) for c, w in zip(correct, wrong)]

    return sum(s) / len(y_target)

def error(y_target, y_pred):

    err = torch.sum(y_target.squeeze() != y_pred)

    return err / len(y_pred)