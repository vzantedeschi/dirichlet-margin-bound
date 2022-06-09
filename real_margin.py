import hydra
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from core.margin_bounds import BOUNDS
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset
from models.majority_vote import MajorityVote
from models.random_forest import decision_trees
from models.stumps import uniform_decision_stumps

def margin_error(w_theta, gamma=0.):
    return (w_theta > 0.5 - gamma).mean()

@hydra.main(config_path='config/grid_search.yaml')
def main(cfg):

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/margin-bounds/from-{cfg.model.base}/{cfg.bound.name}/{cfg.model.pred}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/prior={cfg.model.prior}/grid_size={cfg.training.grid_size}"

    # use <cfg.model.base>'s solution as posterior
    if "margin" in cfg.model.base:
        g = "0.01"
    else: 
        g = "0"

    LOAD_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.model.base}/{cfg.model.pred}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/prior=1/gamma={g}/lr=0.1/batch-size=100/"

    ROOT_DIR = Path(ROOT_DIR)
    LOAD_DIR = Path(LOAD_DIR)

    print("results will be saved in:", ROOT_DIR.resolve()) 

    grid = np.logspace(-4, np.log10(0.5), num=cfg.training.grid_size, endpoint=False)

    if cfg.bound.name == "kth_margin":
        delta = cfg.bound.delta # no union bound
    else:
        delta = cfg.bound.delta / cfg.training.grid_size
    print(delta)
    bound = BOUNDS[cfg.bound.name]

    train_errors, test_errors, margin_errors, bounds, gammas, ks = [], [], [], [], [], []
    for i in range(cfg.num_trials):
        
        print("seed", cfg.training.seed+i)
        deterministic(cfg.training.seed+i)

        SAVE_DIR = ROOT_DIR / f"seed={cfg.training.seed+i}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if (SAVE_DIR / "err-b.npy").is_file():
            print(SAVE_DIR)
            # load saved stats
            seed_results = np.load(SAVE_DIR / "err-b.npy", allow_pickle=True).item()

        else:

            seed_results = {}

            data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)

            n = len(data.X_train)

            if cfg.model.pred == "stumps-uniform":
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

                train = (torch.from_numpy(data.X_train), torch.from_numpy(data.y_train))

            elif cfg.model.pred == "rf": # random forest

                if cfg.model.tree_depth == "None":
                    cfg.model.tree_depth = None

                m_prior = int(n * cfg.model.m_prior)
                n -= m_prior

                predictors, M = decision_trees(cfg.model.M, (data.X_train[:m_prior], data.y_train[:m_prior]), max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth)

                train = (torch.from_numpy(data.X_train[m_prior:]), torch.from_numpy(data.y_train[m_prior:]))

            else:
                raise NotImplementedError("model.pred should be one of the following: [stumps-uniform, rf]")

            # prior
            betas = np.ones(M) * cfg.model.prior # prior

            # load posterior learned with F2 bound
            load_path = LOAD_DIR / f"seed={cfg.training.seed+i}" / "err-b.npy"
            loaded_dict = np.load(load_path, allow_pickle=True).item()
            
            if cfg.model.base in ["f2", "margin"]:
                alphas = loaded_dict['posterior']
                K = alphas.sum()
                theta = alphas / K
            else:
                theta = loaded_dict['posterior']
                K = 1.

            train_err = loaded_dict['train-error']
            test_err = loaded_dict['test-error']
            # print("F2 model stats:", loaded_dict)

            train_preds = predictors(train[0])
            w_theta = torch.where(train[1].unsqueeze(-1) != train_preds, torch.from_numpy(theta), torch.zeros(1)).sum(1).detach().numpy() # ratio of wrong classifiers    

            pbar = tqdm(np.flip(grid))

            best_k, k, best_gamma, best_bound, best_err = K, K, 0.5, np.Inf, 1.
            for gamma in pbar:

                margin_errs = margin_error(w_theta, gamma)
                bound_value = bound(M, gamma, n, margin_errs, delta, K, theta, betas)

                if cfg.bound.name in ["dirichlet_tightest", "dirichlet_gibbs"]:
                    bound_value, k = bound_value.fun, bound_value.x

                if bound_value < best_bound:
                    best_bound = bound_value
                    best_gamma = gamma
                    best_err = margin_errs
                    best_k = k

                pbar.set_description(f"best bound value {best_bound}")

            seed_results |= {
                "train-error": train_err,
                "test-error": test_err,
                "train-risk": best_err,
                "bound": best_bound,
                "posterior": theta,
                "gamma": best_gamma,
                "K": best_k,
            }

            print(f"Best (K, gamma) = ({best_k}, {best_gamma}) gives bound value of {best_bound}")
            # save seed results
            np.save(SAVE_DIR / "err-b.npy", seed_results)

        train_errors.append(seed_results["train-error"])
        test_errors.append(seed_results["test-error"])
        bounds.append(seed_results["bound"])
        margin_errors.append(seed_results["train-risk"])
        gammas.append(seed_results["gamma"])
        ks.append(seed_results["K"])
 
    results = {"train-error": (np.mean(train_errors), np.std(train_errors)), "test-error": (np.mean(test_errors), np.std(test_errors)), "bound": (np.mean(bounds), np.std(bounds)), "train-risk": (np.mean(margin_errors), np.std(margin_errors)), "gamma": (np.mean(gammas), np.std(gammas)), "K": (np.mean(ks), np.std(ks))}

    np.save(ROOT_DIR / "err-b.npy", results)

    print(results)

if __name__ == "__main__":
    main()
