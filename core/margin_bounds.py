import numpy as np

from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import xlogy, loggamma, digamma

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def kl_dirichlet(alphas, betas):

    res = loggamma(alphas.sum()) - loggamma(alphas).sum()
    res -= loggamma(betas.sum()) - loggamma(betas).sum()
    res += np.sum((alphas - betas) * (digamma(alphas) - digamma(alphas.sum())))
    
    return res

def bernoulli_small_kl(q, p):
    return xlogy(q, q/p) + xlogy(1-q, (1-q)/(1-p))

def invert_small_kl(train_loss, rhs):
    "Get the inverted small-kl, largest p such that kl(train_loss : p) \le rhs"
    if train_loss >= 1.:
        return 1.
    start = np.minimum(train_loss + np.sqrt(rhs / 2.), 0.99)    # start at McAllester or at 0.99
    try:
        res = root_scalar(lambda r: bernoulli_small_kl(train_loss, r) - rhs,
                        x0=start, bracket=[train_loss + 1e-8, 1. - 1e-8])
    except ValueError:
        return 1.
    return res.root

def improved_bg_bound(dim, gamma, m, margin_errs, delta, *args):
    "Improved version of bound from B&G 2022"
    T = np.ceil(2 * np.log(m) / (gamma ** 2))
    rhs = (T * np.log(dim) + np.log(2 * np.sqrt(m) / delta)) / m
    return invert_small_kl(margin_errs + 1/m, rhs) + 1/m


def kth_margin_bound(dim, gamma, m, margin_errs, delta, *args):
    "kth margin bound from Gao & Zhou 2013"
    # Note kth margin gives error of k-1, i.e. 1st margin is 0 error.
    q = 2 * np.log(2 * dim) / (gamma ** 2)
    q *= np.log((2 * m ** 2) / np.log(dim))
    q += np.log(m * dim / delta)
    return invert_small_kl(margin_errs, q/m) + np.log(dim) / m

def better_margins_paper_bound(dim, gamma, m, margin_errs, delta, *args):
    "Second improved version of B&G, optimises T."
    def bnd(T):
        T = np.ceil(np.maximum(T, 1.))
        comp = T * np.log(dim) + np.log(m / delta)
        exp = np.exp(- 0.5 * T * (gamma ** 2))
        return invert_small_kl(margin_errs + exp, comp / m) + exp
    return minimize_scalar(bnd, bracket=(2,1e3), method="golden").fun

def better_margins_paper_bound2(dim, gamma, m, margin_errs, delta, _, theta, *args):
    "Second improved version of B&G, optimises T."
    def bnd(T):
        T = np.ceil(np.maximum(T, 1.))
        comp = T * (np.log(dim) + np.sum(theta * np.log(theta))) + np.log(m / delta)
        exp = np.exp(- 0.5 * T * (gamma ** 2))
        return invert_small_kl(margin_errs + exp, comp / m) + exp
    return minimize_scalar(bnd, bracket=(2,1e3), method="golden").fun

def margin_bound(dim, gamma, m, margin_errs, delta, K, theta, betas):
    
    exp_conc = np.exp(- (K + 1) * gamma**2)
    kl_div = kl_dirichlet(K * theta, betas)
    const = kl_div + np.log(2 * (m**0.5) / delta)

    if kl_div <= 0.:
        return 1.

    bound = invert_small_kl(margin_errs + exp_conc, const / m) + exp_conc

    return bound

def margin_gibbs(dim, gamma, m, margin_errs, delta, K, theta, betas):
    
    exp_conc = np.exp(- 4 * (K + 1) * gamma**2)
    kl_div = kl_dirichlet(K * theta, betas)
    const = kl_div + np.log(2 * (m**0.5) / delta)

    if kl_div <= 0.:
        return 1.

    bound = invert_small_kl(margin_errs, const / m) + exp_conc

    return bound

def tightest_margin_bound(dim, gamma, m, margin_errs, delta, K, theta, betas):
    try:
        res = minimize_scalar(lambda r: margin_bound(dim, gamma, m, margin_errs, delta, r, theta, betas),
                              bracket=(K, K * 2**16), method="golden")
    except ValueError:
        return 1.

    return res

def tightest_margin_gibbs(dim, gamma, m, margin_errs, delta, K, theta, betas):
    try:
        res = minimize_scalar(lambda r: margin_gibbs(dim, gamma, m, margin_errs, delta, r, theta, betas),
                              bracket=(K, K * 2**16), method="golden")
    except ValueError:
        return 1.

    return res

BOUNDS = {
    "bg+": improved_bg_bound,
    "gz": kth_margin_bound,
    "dirichlet": tightest_margin_bound,
    "dirichlet_gibbs": tightest_margin_gibbs
}
