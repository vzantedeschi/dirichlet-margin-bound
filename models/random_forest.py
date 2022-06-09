import torch

from sklearn.ensemble import RandomForestClassifier

def trees_predict(x, trees):
    
    pred = torch.stack([torch.from_numpy(t.predict(x)).float() for t in trees], 1)

    return pred

def decision_trees(M, data, max_samples=1., max_features="sqrt", max_depth=None):

    bootstrap = True
    if max_samples == 1.:
        bootstrap = False
        max_samples = None

    forest = RandomForestClassifier(n_estimators=M, criterion="gini", max_depth=max_depth, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples, n_jobs=-1, class_weight='balanced_subsample')
    forest.fit(*data)

    predictors = lambda x: trees_predict(x, forest.estimators_)

    return predictors, M