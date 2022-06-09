from core.bounds import BOUNDS
from core.losses import *

def get_method(cfg, logit_gamma, num_classes=2):

    # define params for each method
    methods = { # type: (loss, bound-coeff, distribution-type, kl factor, bound)


        # margin bound with Seeger's bound backend
        "margin": (
                            lambda x, y, z: margin_loss_binary(x, y, z, logit_gamma=logit_gamma), # train loss
                            lambda x, y, z: margin_loss_binary(x, y, z, logit_gamma=logit_gamma), # test loss
                            1., # bound coefficient
                            "dirichlet", # distribution type
                            1. # KL coefficient
                        ),

        # factor2 bound with Seeger's bound backend
        "f2": (
                        lambda x, y, z: loss_binary(x, y, z), 
                        lambda x, y, z: loss_binary(x, y, z), 
                        2., 
                        "dirichlet", 
                        1.
                    ), 


        # bounds through Gibbs classifier
        "Bin": (
                    lambda x, y, z: rand_loss(x, y, z, n=cfg.training.rand_n), 
                    lambda x, y, z: rand_loss(x, y, z, n=cfg.training.rand_n), 
                    2., 
                    "categorical", 
                    cfg.training.rand_n
                ),

        "FO":   (
                    lambda x, y, z: moment_loss(x, y, z, order=1), 
                    lambda x, y, z: moment_loss(x, y, z, order=1), 
                    2., 
                    "categorical", 
                    1.
                ),

        "SO":   (
                    lambda x, y, z: moment_loss(x, y, z, order=2), 
                    lambda x, y, z: moment_loss(x, y, z, order=2), 
                    4., 
                    "categorical", 
                    2.
                ),
    }

    return methods[cfg.training.risk]