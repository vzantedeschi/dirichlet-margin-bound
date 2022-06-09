import torch
import numpy as np

from torch.distributions.dirichlet import Dirichlet as Dir
from torch import lgamma, digamma

class Dirichlet():

    def __init__(self, alpha):

        self.alpha = alpha

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        res = lgamma(self.alpha.sum()) - lgamma(self.alpha).sum()
        res -= lgamma(beta.sum()) - lgamma(beta).sum()
        res += torch.sum((self.alpha - beta) * (digamma(self.alpha) - digamma(self.alpha.sum())))

        return res

    def mean(self):
        return Categorical(self.alpha.detach())

    def mode(self):
        assert all(self.alpha > 1), "can compute mode only of Dirichlet with alpha > 1"

        alpha = self.alpha - 1

        return alpha / alpha.sum()

    def get_param(self):
        
        return self.alpha

    def entropy(self, of_mean=True):

        if of_mean:
            return self.mean().entropy()
        else:
            return Dir(self.get_param()).entropy()

    def project(self):
        self.alpha.data = self.alpha.data.clamp(min=1e-6)

class DirichletScaled(Dirichlet):

    def __init__(self, alpha, log_scale):

        super(DirichletScaled, self).__init__(alpha)

        self.log_scale = log_scale

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        scaled_alpha = self.alpha * torch.exp(self.log_scale)

        res = lgamma(scaled_alpha.sum()) - lgamma(scaled_alpha).sum()
        res -= lgamma(beta.sum()) - lgamma(beta).sum()
        res += torch.sum((scaled_alpha - beta) * (digamma(scaled_alpha) - digamma(scaled_alpha.sum())))

        return res

    def get_param(self):
        
        return self.alpha * torch.exp(self.log_scale)

    def project(self):
        pass

class Categorical():

    def __init__(self, theta):
        self.theta = theta
        self.theta.data = torch.log(theta.data)
        
    def KL(self, beta):

        t = self.get_param()

        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()

    def get_param(self):
        
        return torch.softmax(self.theta, dim=0)

    def entropy(self):

        theta = self.get_param()
        return - torch.sum(theta * torch.log(theta))

    def project(self):
        pass

distr_dict = {
    "dirichlet": Dirichlet,
    "categorical": Categorical,
    "dirichlet-scaled": DirichletScaled
}

if __name__ == "__main__":

    distr = DirichletScaled(torch.tensor([1., 1., 1000.]), torch.log(torch.tensor(100)))
    print(distr.KL(torch.tensor([3., 1., 1.])))