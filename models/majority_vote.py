import numpy as np
import torch

from torch.nn.functional import one_hot

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, distr="dirichlet", kl_factor=1., init_post=True):

        super(MajorityVote, self).__init__()
        
        if distr not in ["dirichlet", "categorical"]:
            raise NotImplementedError

        assert all(prior > 0), "all prior parameters must be positive"
        self.num_voters = len(prior)

        self.prior = prior
        self.voters = voters
        self.distr_type = distr

        if init_post:
            post = torch.rand(self.num_voters) * 2 + 1e-9 # uniform draws in (0, 100]
            self.post = torch.nn.Parameter(post, requires_grad=True)
            self.distribution = distr_dict[distr](self.post)

        self.kl_factor = kl_factor

    def forward(self, x):
        return self.voters(x)

    def voter_strength(self, data):
        """ expected accuracy of a voter of the ensemble"""
        # import pdb; pdb.set_trace()
        
        y_target, y_pred = data

        l = torch.where(y_target == y_pred, torch.tensor(1.), torch.tensor(0.))

        return l.mean(1)

    def predict(self, X, num_classes=2): # prediction of deterministic model
        
        if self.distr_type == "dirichlet":

            expected_model = self.distribution.mean()
            theta = expected_model.get_param()

        else:
            theta = self.distribution.get_param()

        voter_preds = self(X).long()

        y_pred = one_hot(voter_preds, num_classes=num_classes).transpose(1, 0).float() # [M, n]

        labels = torch.einsum('i,ijk->jk', theta, y_pred)

        labels = torch.argmax(labels, 1)

        return labels 

    def KL(self):

        return self.kl_factor * self.distribution.KL(self.prior)

    def get_post(self):
        return self.distribution.get_param()

    def get_post_grad(self):
        return self.post.grad

    def set_post(self, value):

        assert all(value > 0), "all posterior parameters must be positive"
        assert len(value) == self.num_voters
         
        if self.distr_type == "categorical": # make sure params sum to 1
            value /= value.sum()

        self.post = torch.nn.Parameter(value, requires_grad=True)

        self.distribution.alpha = self.post

    def entropy(self):
        return self.distribution.entropy()

    def project(self):
        self.distribution.project()


class MajorityVoteScaled(MajorityVote):

    def __init__(self, voters, post, prior, kl_factor=1.):

        super(MajorityVoteScaled, self).__init__(voters, prior, distr="dirichlet", kl_factor=kl_factor, init_post=False)

        self.post = post
        self.log_scale = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.distribution = distr_dict["dirichlet-scaled"](self.post, self.log_scale)

    def get_post_grad(self):
        return self.log_scale.grad
