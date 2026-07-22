"""Clean-room SGD uncertainty quantification from "Accurate Large-sample Uncertainty Quantification
using Stochastic Gradient Descent" (arXiv 2606.00293). numpy, CPU.
c1: relative covariance error ~ sqrt(stepsize). c2: exact minibatch noise cov.
c3: W2 between true and proxy distributions. c4: two-stage tuning.
"""
from __future__ import annotations
import numpy as np


def sgd_steady_cov(grad_fn, x_star, alpha, sigma, n_samples, burn_in, thin, seed=0):
    """Run constant-stepsize SGD, collect steady-state samples, return covariance."""
    rng = np.random.default_rng(seed); d = len(x_star); x = np.zeros(d)
    for _ in range(burn_in):
        x = x - alpha * (grad_fn(x) + sigma * rng.standard_normal(d))
    samples = np.empty((n_samples, d))
    for k in range(n_samples):
        for _ in range(thin):
            x = x - alpha * (grad_fn(x) + sigma * rng.standard_normal(d))
        samples[k] = x
    return np.cov(samples.T) + 1e-10 * np.eye(d), samples


def proxy_cov(alpha, grad_fn, x_star, sigma):
    """Discrete-time proxy covariance: solve Lyapunov equation."""
    d = len(x_star); H = np.zeros(d)
    eps = 1e-5
    for i in range(d):
        ei = np.zeros(d); ei[i] = eps
        g_plus = grad_fn(x_star + ei); g_minus = grad_fn(x_star - ei); H[i] = (g_plus[i] - g_minus[i]) / (2 * eps)
    # Lyapunov: (I - alpha*diag(H)) Sigma + Sigma (I - alpha*diag(H))^T = alpha^2 * sigma^2 * I
    A = (np.eye(d) - alpha * np.diag(H))
    Q = alpha ** 2 * sigma ** 2 * np.eye(d)
    # Solve A Sigma + Sigma A^T = Q (vectorized)
    d2 = d * d
    M = np.kron(A, np.eye(d)) + np.kron(np.eye(d), A)
    sigma_vec = np.linalg.solve(M, Q.flatten())
    return sigma_vec.reshape(d, d)


def minibatch_noise_cov(X, batch_size, grad_fn, x, seed=0):
    """Exact minibatch noise covariance (no constant-noise assumption, Theorem 4.3)."""
    rng = np.random.default_rng(seed); n, d = X.shape
    grads = np.array([grad_fn(X[i]) for i in range(n)])
    mean_grad = grads.mean(0)
    # covariance of a random minibatch gradient
    cov = np.zeros((d, d))
    n_trials = 200
    for _ in range(n_trials):
        idx = rng.choice(n, batch_size, replace=False)
        batch_grad = grads[idx].mean(0)
        diff = (batch_grad - mean_grad).reshape(-1, 1)
        cov += diff @ diff.T
    return cov / n_trials


def w2_distance(samples1, samples2):
    """2-Wasserstein distance between two empirical distributions (Gaussian approximation)."""
    mu1, Sig1 = samples1.mean(0), np.cov(samples1.T) + 1e-8 * np.eye(samples1.shape[1])
    mu2, Sig2 = samples2.mean(0), np.cov(samples2.T) + 1e-8 * np.eye(samples2.shape[1])
    Sig1_sqrt = np.linalg.cholesky(Sig1)
    cross = Sig1_sqrt.T @ Sig2 @ Sig1_sqrt
    eigvals = np.maximum(np.real(np.linalg.eigvalsh(cross)), 0)
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(Sig1) + np.trace(Sig2) - 2 * np.sum(np.sqrt(eigvals)))
