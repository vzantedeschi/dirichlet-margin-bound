import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def load_normals(n_train, n_test, means, scales):

    train_x, train_y = [], []
    test_x, test_y = [], []

    for i, (m, s) in enumerate(zip(means, scales)):
        train_x.append(np.random.multivariate_normal(m, s, n_train))
        test_x.append(np.random.multivariate_normal(m, s, n_test))

        train_y.append(i * np.ones(n_train))
        test_y.append(i * np.ones(n_test))

    train_x, train_y, test_x, test_y = np.vstack(train_x), np.concatenate(train_y), np.vstack(test_x), np.concatenate(test_y)

    return train_x, train_y, test_x, test_y
    
def load_moons(n_train, n_test, noise=0.05, normalize=True):
    
    train_c1 = np.random.binomial(n_train, 0.5) # draw number of train points for class 1
    test_c1 = np.random.binomial(n_test, 0.5) # draw number of test points for class 1
    
    train_x, train_y = datasets.make_moons(n_samples=(train_c1, n_train - train_c1), noise=noise)
    test_x, test_y = datasets.make_moons(n_samples=(test_c1, n_test - test_c1), noise=noise)

    if normalize:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    return train_x, train_y, test_x, test_y

def load_blobs(n_train, n_test, num_classes=4, normalize=True):
    
    x, y = datasets.make_blobs(n_samples=n_train+n_test, centers=num_classes)
    train_x, test_x, train_y, test_y = x[:n_train], x[n_train:], y[:n_train], y[n_train:]

    if normalize:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    return train_x, train_y, test_x, test_y

DATASETS = {
    "normals": load_normals,
    "moons": load_moons,
    "blobs": load_blobs,
}

def toy_dataset(name, n_train, n_test, **kwargs):

    if name == "normals":
        X_train, y_train, X_test, y_test = DATASETS[name](n_train // 2, n_test // 2, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

    elif name == "moons":
        noise = kwargs.pop("noise", 0.05)
        X_train, y_train, X_test, y_test = DATASETS[name](n_train, n_test, noise)

    else:
        num_classes = kwargs.pop("num_classes", 4)
        X_train, y_train, X_test, y_test = DATASETS[name](n_train, n_test, num_classes=num_classes)
    
    X_train, y_train = shuffle(X_train, y_train)
    
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
