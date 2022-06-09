import numpy as np
import torch

from core.kl_inv import klInvFunction

def order_magnitude(f):
    return 10 ** (- torch.floor(torch.log10(f))) / 2

def mcallester_bound(n, model, risk, delta, coeff=1, verbose=False, monitor=None):

    kl = model.KL()

    const = np.log(2 * (n**0.5) / delta)

    bound = coeff * (risk + ((kl + const) / 2 / n)**0.5)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound 

def seeger_bound(n, model, risk, delta, coeff=1, verbose=False, monitor=None):

    kl = model.KL()

    const = np.log(2 * (n**0.5) / delta)

    bound = coeff * klInvFunction.apply(risk, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Seeger bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound 

def margin_bound(n, model, risk, delta, coeff=1, verbose=False, monitor=None, logit_gamma=0.1, gamma_learned=False, inner_bound=seeger_bound):

    alpha = model.get_post()

    gamma = torch.sigmoid(logit_gamma) / 2

    if gamma_learned: # because of union bound over different gammas
        delta /= order_magnitude(gamma).detach().numpy()
    
    bound = inner_bound(n, model, risk, delta, coeff, verbose, monitor)

    exp_conc = torch.exp(- 4 * (alpha.sum() + 1) * gamma**2)

    if verbose:
        print(f"alpha_0={alpha.sum()}, exp_concentration={exp_conc}, gamma={gamma}, d={len(alpha)}, n={n}")
        print(f"Margin bound={(bound + exp_conc).item()}\n")

    if monitor:
        monitor.write(train={"margin-risk": risk, "exp_concentration": exp_conc, "gamma": gamma})

    return bound + exp_conc

BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound,
    "margin": margin_bound
}
