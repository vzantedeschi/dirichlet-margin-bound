import hydra
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.bounds import BOUNDS
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MajorityVote
from models.random_forest import decision_trees
from models.stumps import uniform_decision_stumps
from optimization import stochastic_routine
from config_utils import get_method

@hydra.main(config_path='config/real.yaml')
def main(cfg):

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.model.pred}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/prior={cfg.model.prior}/gamma={cfg.training.gamma}/lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/"

    ROOT_DIR = Path(ROOT_DIR)

    print("results will be saved in:", ROOT_DIR.resolve()) 

    if cfg.training.gamma == "None":
        logit_gamma = torch.nn.Parameter(torch.tensor(-4.), requires_grad=True) # later apply sigmoid(gamma)/2 to enforce gamma in (0, 0.5)
    else:
        logit_gamma = torch.logit(torch.tensor(cfg.training.gamma * 2))

    train_errors, test_errors, train_losses, bounds, strengths, entropies, kls, times = [], [], [], [], [], [], [], []
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
            
            # get method tools
            loss, test_loss, bound_coeff, distr, kl_factor = get_method(cfg, logit_gamma, num_classes=data.num_classes)

            if cfg.model.pred == "stumps-uniform":
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

                train = TorchDataset(data.X_train, data.y_train)

            elif cfg.model.pred == "rf": # random forest

                if cfg.model.tree_depth == "None":
                    cfg.model.tree_depth = None

                m_prior = int(n * cfg.model.m_prior)
                n -= m_prior

                predictors, M = decision_trees(cfg.model.M, (data.X_train[:m_prior], data.y_train[:m_prior]), max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth)

                train = TorchDataset(data.X_train[m_prior:], data.y_train[m_prior:])

            else:
                raise NotImplementedError("model.pred should be one of the following: [stumps-uniform, rf]")

            if cfg.training.risk.startswith("margin"):
                bound = lambda _, model, risk: BOUNDS["margin"](n, model, risk, delta=cfg.bound.delta, coeff=bound_coeff, logit_gamma=logit_gamma, monitor=monitor)

            else:
                bound = lambda _, model, risk: BOUNDS["seeger"](n, model, risk, delta=cfg.bound.delta, coeff=bound_coeff, monitor=monitor)                

            trainloader = DataLoader(train, batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)
            testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096, num_workers=cfg.num_workers, shuffle=False)

            betas = torch.ones(M) * cfg.model.prior # prior

            model = MajorityVote(predictors, betas, distr=distr, kl_factor=kl_factor)

            monitor = MonitorMV(SAVE_DIR)
            optimizer = Adam(list(model.parameters()) + [logit_gamma], lr=cfg.training.lr)
            # init learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

            best_model, _, best_train_stats, best_test_stats, time = stochastic_routine(trainloader, testloader, model, optimizer, bound=bound, loss=loss, test_loss=test_loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler, num_classes=data.num_classes)
        
            seed_results |= {
                "train-error": best_train_stats['error'],
                "test-error": best_test_stats['error'],
                "train-risk": best_train_stats["loss"],
                "bound": best_train_stats["bound"],
                "time": time,
                "posterior": best_model.get_post().detach().numpy(),
                "strength": best_train_stats["strength"],
                "KL": best_model.KL().item(),
                "entropy": best_model.entropy().item(),
            }
            print(f"train error: {seed_results['train-error']}, test error: {seed_results['test-error']}")

            if cfg.training.risk.startswith("margin"):
                b = float(BOUNDS["margin"](n, best_model, torch.tensor(best_train_stats["loss"]), delta=cfg.bound.delta, coeff=bound_coeff, logit_gamma=logit_gamma, verbose=True))

            else:
                b = float(BOUNDS["seeger"](n, best_model, torch.tensor(best_train_stats["loss"]), delta=cfg.bound.delta, coeff=bound_coeff, verbose=True))

            # save seed results
            np.save(SAVE_DIR / "err-b.npy", seed_results)
            monitor.close()

        train_errors.append(seed_results["train-error"])
        test_errors.append(seed_results["test-error"])
        entropies.append(seed_results["entropy"])
        strengths.append(seed_results["strength"])
        kls.append(seed_results["KL"])
        bounds.append(seed_results["bound"])
        times.append(seed_results["time"])
        train_losses.append(seed_results["train-risk"])
 
    results = {"train-error": (np.mean(train_errors), np.std(train_errors)), "test-error": (np.mean(test_errors), np.std(test_errors)), "bound": (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times)), "strength": (np.mean(strengths), np.std(strengths)), "train-risk": (np.mean(train_losses), np.std(train_losses)), "entropy": (np.mean(entropies), np.std(entropies)), "KL": (np.mean(kls), np.std(kls))}

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()
