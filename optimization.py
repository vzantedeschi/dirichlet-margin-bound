import torch

from copy import deepcopy
from time import time
from tqdm import tqdm

from core.losses import error

def train_batch(data, model, optimizer, loss, bound=None, nb_iter=1e4, monitor=None):

    model.train()

    pbar = tqdm(range(int(nb_iter)))
    no_improv = 0
    best_cost = torch.tensor(10.)
    for i in pbar:
        optimizer.zero_grad()

        risk = loss(*data, model.get_post())

        if bound is not None:
            cost = bound(model, risk)

        else:
            cost = risk

        pbar.set_description("train obj %s" % cost.item())

        cost.backward()
        optimizer.step()
     
        model.project()
        
        if best_cost < cost:
            no_improv += 1
        else:
            best_cost = cost
            no_improv = 0

        if no_improv == 50:
            break

        if monitor:
            monitor.write_all(i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})

def train_stochastic(dataloader, model, optimizer, loss, epoch, bound=None, monitor=None):

    model.train()

    last_iter = epoch * len(dataloader)

    pbar = tqdm(dataloader)

    for i, batch in enumerate(pbar):

        n = len(batch[0])
        data = batch[1], model(batch[0])

        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        
        risk = loss(*data, model.get_post())

        if bound is not None:
            cost = bound(n, model, risk)

        else:
            cost = risk
            
        pbar.set_description("avg train obj %f" % (cost.item()))

        cost.backward()
        optimizer.step()

        model.project()
        
        if monitor:
            monitor.write_all(last_iter+i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})
            
def evaluate(dataloader, model, loss, epoch=-1, bounds=None, monitor=None, tag="val", num_classes=2):
    """ evaluate error, loss, optional bounds based on loss"""

    model.eval()

    risk = 0.
    err = 0.
    strength = 0.
    n = 0

    for x, y in dataloader:

        model_preds = model.predict(x, num_classes)
        voter_preds = model(x)

        err += error(y, model_preds) * len(x)
        risk += loss(y, voter_preds, model.get_post()) * len(x)
        strength += sum(model.voter_strength((y, voter_preds)))

        n += len(x)

    risk /= n
    err /= n
    strength /= n

    total_metrics = {"error": err.item(), "loss": risk.item(), "strength": strength.item()}

    if bounds is not None:

        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics

def stochastic_routine(trainloader, testloader, model, optimizer, loss, bound, test_loss=None, monitor=None, num_epochs=100, lr_scheduler=None, num_classes=2):

    best_bound = float("inf")
    best_e = -1
    no_improv = 0
    best_train_stats = None

    test_loss = loss if test_loss is None else test_loss
    
    t1 = time()
    for e in range(num_epochs):
        train_stochastic(trainloader, model, optimizer, loss, epoch=e, bound=bound, monitor=monitor)

        train_stats = evaluate(trainloader, model, test_loss, epoch=e, bounds={"bound": bound}, monitor=monitor, tag="train", num_classes=num_classes)
        # print(f"Epoch {e}: {train_stats['bound']}\n")
        
        no_improv += 1
        if train_stats["bound"] < best_bound:
            best_bound = train_stats["bound"]
            best_train_stats = train_stats
            best_e = e
            best_model = deepcopy(model)
            no_improv = 0

        # reduce learning rate if needed
        if lr_scheduler:
            lr_scheduler.step(train_stats["bound"])

        if no_improv == max(2, num_epochs // 4):
            break

    t2 = time()

    best_test_stats = evaluate(testloader, best_model, test_loss, num_classes=num_classes)

    # print(f"Test error: {test_error['error']}; bound: {best_train_stats["bound"]}\n")

    return best_model, best_bound, best_train_stats, best_test_stats, t2-t1
