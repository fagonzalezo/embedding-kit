"""Dataset builders for the EmbedKit pilot (no downloads)."""
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def load_pets():
    d = np.load(REPO / "examples" / "data" / "pets_resnet50.npz")
    return (d["X_train"].astype(np.float32), d["y_train"].astype(np.int64),
            d["X_test"].astype(np.float32), d["y_test"].astype(np.int64))


def load_digits_split(seed=0):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    return Xtr.astype(np.float32), ytr.astype(np.int64), Xte.astype(np.float32), yte.astype(np.int64)


def make_synthetic(n=6000, n_classes=10, latent=16, D=512, sep=1.0, hidden=128,
                   mean_scale=20.0, nuis_scale=6.0, n_nuis=3, seed=0):
    """10 Gaussian classes on a 16-d latent -> fixed random 2-layer tanh MLP -> 512-d,
    + large shared mean vector + 3 high-variance class-independent nuisance directions."""
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(n_classes, latent))
    y = np.repeat(np.arange(n_classes), n // n_classes)
    Z = centers[y] + rng.normal(0, 1.0, size=(n, latent))
    W1 = rng.normal(0, 1 / np.sqrt(latent), size=(latent, hidden)); b1 = rng.normal(0, 0.1, hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, D)); b2 = rng.normal(0, 0.1, D)
    H = np.tanh(Z @ W1 + b1)
    X = np.tanh(H @ W2 + b2)
    sig_std = X.std(axis=0).mean()
    # shared mean vector: norm = mean_scale * typical per-sample deviation norm
    dev_norm = np.linalg.norm(X - X.mean(0), axis=1).mean()
    mu = rng.normal(size=D); mu = mu / np.linalg.norm(mu) * mean_scale * dev_norm
    # nuisance: orthonormal directions with per-direction std = nuis_scale * dev_norm/sqrt(n_nuis)
    Q, _ = np.linalg.qr(rng.normal(size=(D, n_nuis)))
    coef = rng.normal(0, nuis_scale * dev_norm / np.sqrt(n_nuis), size=(n, n_nuis))
    X = X + mu + coef @ Q.T
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    info = dict(sig_std=float(sig_std), dev_norm=float(dev_norm), mean_norm=float(np.linalg.norm(mu)),
                sep=sep, mean_scale=mean_scale, nuis_scale=nuis_scale)
    return Xtr.astype(np.float32), ytr, Xte.astype(np.float32), yte, info
