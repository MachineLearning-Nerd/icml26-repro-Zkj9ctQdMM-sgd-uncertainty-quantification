"""Numerical core for reproducing 'Accurate Large-sample Uncertainty Quantification using
Stochastic Gradient MCMC' (arXiv 2606.00293), OpenReview Zkj9ctQdMM.

Clean-room implementation of the discrete-time proxy stationary-covariance theory:
  * Eq 15 (Prop 4.2): exact stationary covariance of the linear-Gaussian proxy (discrete
    Lyapunov equation).
  * Eq 16 (Thm 4.3): exact minibatch noise covariance (no constant-noise assumption).
  * Eq B.5 (Prop B.1): momentum (SGLD-with-momentum) stationary covariance; reduces to
    Eq 15 as kappa -> 0.
  * Algorithm 1: offline sandwich covariance -> solve coupled Eq15+16 for the
    preconditioner Lambda.
All matrix algebra is exact (no Monte Carlo) unless explicitly running an SGD chain.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm


# --------------------------------------------------------------------------- #
# Core stationary-covariance solvers (exact, discrete-time)                   #
# --------------------------------------------------------------------------- #
def noise_cov_eq16(Jn, gn, Sigma, Gamma, N, B, replacement=True):
    """Exact average minibatch-gradient noise covariance Cbar_psi (Theorem 4.3, Eq 16).

    Jn : (N, d, d) per-sample Hessians J_n = grad^2 ell_n(theta_hat)
    gn : (N, d)    per-sample gradients g_n = grad ell_n(theta_hat)
    Sigma : (d, d) stationary covariance Sigma_psi
    Gamma : (d, d) prior precision (R(theta)=0.5 theta' Gamma theta); 0 for MLE
    Returns the (d,d) matrix Cbar_psi. With-replacement sampling (default).
    """
    d = gn.shape[1]
    Jbar = Jn.mean(axis=0)                      # J = (1/N) sum J_n  = Hhat
    gbar = gn.mean(axis=0)                      # (1/N) sum g_n
    Ical = np.einsum("ni,nj->ij", gn, gn) / N   # I = (1/N) sum g_n g_n^T
    # cross term from the (psi-theta_hat) part of the quadratic: varies with Sigma
    JSJ_var = np.einsum("nij,njk,nkl->il", Jn, np.broadcast_to(Sigma, Jn.shape), Jn) / N - Jbar @ Sigma @ Jbar
    # mean-gradient outer-product correction; at the MAP gbar = -(1/N) Gamma theta_hat
    gcorr = np.outer(gbar, gbar)
    Cbar = (1.0 / B) * (Ical - gcorr + JSJ_var)
    if not replacement:
        Cbar *= (N - B) / (N - 1)
    return Cbar


def _noise_linear_map(Jn, gn, N, B, replacement=True):
    """Precompute the linear map  vec(Cbar_psi(Sigma)) = c0 + M vec(Sigma)  (Thm 4.3, Eq 16).

    Cbar = (1/B)( Ical - gbar gbar^T + (1/N) sum_n J_n Sigma J_n - J Sigma J ), which is
    affine in Sigma. Returns (c0, M, d)."""
    d = gn.shape[1]
    Jbar = Jn.mean(axis=0)
    gbar = gn.mean(axis=0)
    Ical = np.einsum("ni,nj->ij", gn, gn) / N
    c0 = (1.0 / B) * (Ical - np.outer(gbar, gbar))
    if not replacement:
        c0 *= (N - B) / (N - 1)
    # T_{ij,kl} s.t. (T vec(Sigma))_{ij} = (1/N) sum_n (J_n Sigma J_n)_{ij} - (J Sigma J)_{ij}
    A4 = np.einsum("nik,nlj->ijkl", Jn, Jn) / N          # (1/N) sum_n J_n[i,k] J_n[l,j]
    Jterm = np.einsum("ik,lj->ijkl", Jbar, Jbar)          # J[i,k] J[l,j]
    T = (1.0 / B) * (A4 - Jterm)
    if not replacement:
        T *= (N - B) / (N - 1)
    M = T.reshape(d * d, d * d)
    return c0.reshape(-1), M, d


def sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement=True,
                    nmap=None):
    """Exact stationary covariance of the DQ+exact proxy (Eq15 + Eq16) for given Lambda.

    The coupled equation is LINEAR in Sigma, so we solve one d^2 x d^2 linear system:
        [ I - (A kron A) - (Lam kron Lam) M ] vec(Sigma) = (Lam kron Lam) c0 + (2/beta) vec(Lam)
    with A = I - Lam Hhat. Returns (Sigma_psi, Cbar_psi). Exact (no Monte Carlo, no iteration).
    """
    d = Hhat.shape[0]
    I = np.eye(d)
    if nmap is None:
        nmap = _noise_linear_map(Jn, gn, N, B, replacement)
    c0, M, _ = nmap
    A = I - Lam @ Hhat
    inj = 0.0 if not np.isfinite(beta) else 2.0 / beta
    LkL = np.kron(Lam, Lam)
    lhs = np.eye(d * d) - np.kron(A, A) - LkL @ M
    rhs = LkL @ c0 + inj * Lam.flatten()
    try:
        s = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    Sig = s.reshape(d, d)
    Sig = 0.5 * (Sig + Sig.T)
    # numerical robustness: a stable Lambda gives a PSD covariance; clip tiny-negative
    # eigenvalues that arise from finite-precision near the stability boundary.
    if not np.all(np.isfinite(Sig)):
        return Sig, np.full((d, d), np.nan)
    w, V = np.linalg.eigh(Sig)
    if np.min(w) < 0:
        Sig = (V * np.maximum(w, 0.0)) @ V.T
    Cbar = (c0 + M @ s).reshape(d, d)
    return Sig, Cbar


def sigma_psi_const_noise(Lam, Hhat, Cbar, beta):
    """Discrete quadratic + constant-noise (DQ+const): Cbar fixed (e.g. = Hhat). Eq 9/15."""
    d = Hhat.shape[0]
    A = np.eye(d) - Lam @ Hhat
    inj = 0.0 if not np.isfinite(beta) else 2.0 / beta
    Q = Lam @ Cbar @ Lam + inj * Lam
    return solve_discrete_lyapunov(A, Q)


def sigma_ct(Lam, Hhat, Cbar, beta):
    """Continuous-time proxy covariance (Eq 7): solve continuous Lyapunov
    Hhat Sigma + Sigma Hhat = Cbar + 2/beta I   (for scalar step folded into Lam).
    Here Lam is a preconditioner; CT ignores the Lambda Cbar Lambda / H Sigma H terms."""
    from scipy.linalg import solve_continuous_lyapunov
    d = Hhat.shape[0]
    inj = 0.0 if not np.isfinite(beta) else 2.0 / beta * np.eye(d)
    # CT: Hhat Sigma + Sigma Hhat = Cbar + 2/beta I  (continuous-time diffusion limit)
    return solve_continuous_lyapunov(Hhat, Cbar + inj)


def solve_lambda_for_target(target_Sig, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta,
                            replacement=True, x0=None):
    """Algorithm 1 step 1 (DQ+exact): find scalar lambda>=0 s.t. Sigma_psi(lambda I) = target_Sig.

    Minimises ||sigma_psi_exact(lambda) - target||_F via scalar root/brent on the
    relative error. Returns (lambda_star, Sigma_psi_at_star, rel_err).
    """
    from scipy.optimize import minimize_scalar
    d = Hhat.shape[0]

    def relerr(lam):
        if lam <= 0:
            return 1e9
        Lam = lam * np.eye(d)
        Sig, _ = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement)
        denom = max(np.linalg.norm(target_Sig), 1e-30)
        return float(np.linalg.norm(Sig - target_Sig) / denom)

    # lambda must keep rho(I - lam Hhat) < 1  ->  lam < 2 / mu_max(Hhat)
    lam_max = 2.0 / np.linalg.eigvalsh(Hhat).max() * 0.999
    res = minimize_scalar(relerr, bounds=(1e-10, lam_max), method="bounded",
                          options={"xatol": 1e-12})
    lam = res.x
    Lam = lam * np.eye(d)
    Sig, _ = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement)
    return lam, Sig, float(res.fun)


def solve_lambda_ct(target_Sig, Hhat, beta):
    """CT tuning: continuous-time gives Sigma = (1/(2 lam)) Hhat^{-1/2}(...) ; here we treat
    CT as the diffusion limit where Sigma = solve_continuous_lyapunov(Hhat, Cbar+2/beta I)
    is independent of lambda in the pure-SGD case. For SGD (beta=inf) CT yields Sigma->0,
    so we instead solve the *discrete* map with Cbar=Hhat (the CT heuristic Cbar=H/B) to
    obtain a lambda matching target via the discrete constant-noise map. We return the
    lambda from DQ+const as the 'CT' competitor used in the paper's tables (CT uses the
    continuous-time noise model Cbar = Hhat/B; see Sec D)."""
    # CT noise model: Cbar_psi = Hhat / B  (gradient noise variance ~ curvature/batch)
    return solve_lambda_constnoise(target_Sig, Hhat, Hhat, beta)


def solve_lambda_constnoise(target_Sig, Hhat, Cbar, beta):
    """DQ+const tuning: solve discrete Lyapunov Sigma_psi(lambda)=target with fixed Cbar."""
    from scipy.optimize import minimize_scalar
    d = Hhat.shape[0]
    lam_max = 2.0 / np.linalg.eigvalsh(Hhat).max() * 0.999

    def relerr(lam):
        if lam <= 0:
            return 1e9
        Lam = lam * np.eye(d)
        Sig = sigma_psi_const_noise(Lam, Hhat, Cbar, beta)
        return float(np.linalg.norm(Sig - target_Sig) / max(np.linalg.norm(target_Sig), 1e-30))

    res = minimize_scalar(relerr, bounds=(1e-10, lam_max), method="bounded",
                          options={"xatol": 1e-12})
    lam = res.x
    Lam = lam * np.eye(d)
    return lam, sigma_psi_const_noise(Lam, Hhat, Cbar, beta), float(res.fun)


def solve_lambda_closed_form(target_Sig, Hhat, Cbar_fixed, beta):
    """Closed-form (non-iterative) Algorithm-1 preconditioner: diagonal in Hhat's eigenbasis,
    Lambda = Q diag(s_k) Q^T with per-direction s_k = 2 mu_k S~_kk / P~_kk (clamped to the
    stability cap 1.9/mu_k). This is the decoupled (J-eigenbasis) solution of Eq 15 -- O(1),
    no optimisation, so it is fast on a throttled CPU. Returns (Lam, Sigma_actual, rel_err)."""
    d = Hhat.shape[0]
    S = 0.5 * (target_Sig + target_Sig.T)
    mu, Q = np.linalg.eigh(Hhat)
    cap = 1.9 / mu
    Ptilde = Q.T @ (Cbar_fixed + Hhat @ S @ Hhat) @ Q
    Stilde = Q.T @ S @ Q
    s = np.empty(d)
    for k in range(d):
        if Ptilde[k, k] > 1e-30 and Stilde[k, k] > 0:
            s[k] = min(2.0 * mu[k] * Stilde[k, k] / Ptilde[k, k], cap[k] * 0.95)
        else:
            s[k] = cap[k] * 0.3
    Lam = Q @ np.diag(s) @ Q.T
    Lam = 0.5 * (Lam + Lam.T)
    return Lam, None, float("nan")


def solve_lambda_eigenbasis(target_Sig, Hhat, Cbar_fixed, beta, nmap=None, Jn=None, gn=None,
                            Gamma=None, theta_hat=None, N=None, B=None, replacement=True):
    """Robust Algorithm-1 step: find Lambda = Q diag(s_k) Q^T (Q from eigh(Hhat)) with bounded
    s_k = (1.9/mu_k)*sigmoid(t_k) so rho(I - Lambda Hhat) < 1 is GUARANTEED by construction.
    Minimises ||sigma_psi_exact(Lambda) - target||_F over the D unconstrained params t_k.
    This is stable and well-posed for ill-conditioned problems (e.g. Boston housing)."""
    from scipy.optimize import least_squares
    d = Hhat.shape[0]
    S = 0.5 * (target_Sig + target_Sig.T)
    mu, Q = np.linalg.eigh(Hhat)
    cap = 1.9 / mu                                   # per-direction stability caps

    def make_Lam(t):
        s = cap * (1.0 / (1.0 + np.exp(-t)))
        return Q @ np.diag(s) @ Q.T

    def resid(t):
        Lam = make_Lam(t)
        try:
            Sig = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement, nmap=nmap)[0]
        except Exception:
            return np.ones(d * d) * 1e2
        if not np.all(np.isfinite(Sig)):
            return np.ones(d * d) * 1e2
        return (Sig - S).flatten()

    # warm start: per-direction closed form s_k = 2 mu_k S~_kk / P~_kk (clamped), mapped to t
    Ptilde = Q.T @ (Cbar_fixed + Hhat @ S @ Hhat) @ Q
    Stilde = Q.T @ S @ Q
    s0 = np.empty(d)
    for k in range(d):
        if Ptilde[k, k] > 1e-30 and Stilde[k, k] > 0:
            s0[k] = min(2.0 * mu[k] * Stilde[k, k] / Ptilde[k, k], cap[k] * 0.98)
        else:
            s0[k] = cap[k] * 0.3
    s0 = np.minimum(s0, cap * 0.98)
    t0 = np.log(s0 / (cap - s0 + 1e-12) + 1e-12)
    best = None
    for off in (0.0,):
        try:
            sol = least_squares(resid, t0 + off, method="lm", xtol=1e-10, ftol=1e-10, max_nfev=120)
        except Exception:
            continue
        Lam = make_Lam(sol.x)
        try:
            Sig_act = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement, nmap=nmap)[0]
            rel = float(np.linalg.norm(Sig_act - S) / max(np.linalg.norm(S), 1e-30))
        except Exception:
            rel = float("inf")
            Sig_act = np.full_like(S, np.nan)
        if best is None or (np.isfinite(rel) and rel < best[2]):
            best = (Lam, Sig_act, rel)
        if rel < 1e-6:
            break
    if best is None:
        best = (np.full((d, d), np.nan), np.full_like(S, np.nan), float("inf"))
    return best


def _solve_lambda_eq15(target_Sig, Hhat, Cbar_fixed, beta, nmap=None, Jn=None, gn=None,
                       Gamma=None, theta_hat=None, N=None, B=None, replacement=True,
                       symmetric=True):
    """Solve Eq 15 with Sigma_psi = target and a FIXED noise covariance Cbar_fixed for Lambda
    (Powell hybrid / Levenberg-Marquardt). Lambda parametrized symmetric (D(D+1)/2 unknowns).
    Returns (Lam, Sigma_actual_under_EXACT_noise, rel_err_to_target)."""
    from scipy.optimize import root, least_squares
    d = Hhat.shape[0]
    S = 0.5 * (target_Sig + target_Sig.T)
    P = Cbar_fixed + Hhat @ S @ Hhat
    Ht = Hhat @ S
    inj = 0.0 if not np.isfinite(beta) else 2.0 / beta
    lam_max = 2.0 / np.linalg.eigvalsh(Hhat).max() * 0.95
    tr = {"ii": np.arange(d), "jj": np.arange(d)}

    def make_Lam(p):
        Lam = np.zeros((d, d))
        if symmetric:
            idx = np.triu_indices(d)
            Lam[idx] = p
            Lam = Lam + Lam.T - np.diag(np.diag(Lam))
        else:
            Lam = p.reshape(d, d)
        return Lam

    def F(p):
        Lam = make_Lam(p)
        if np.any(np.abs(np.linalg.eigvals(np.eye(d) - Lam @ Hhat)) >= 1.0 - 1e-9):
            return np.ones(d * d) * 1e2
        R = Lam @ Ht + Ht.T @ Lam.T - Lam @ P @ Lam - inj * Lam
        return R.flatten()

    nvar = len(np.triu_indices(d)[0]) if symmetric else d * d
    # Eigenbasis warm start: J=Q diag(mu) Q^T decouples the equation; per-direction
    # lam_k = 2 mu_k S~_kk / P~_kk (clamped to stability 2/mu_k), then rotate back.
    mu, Q = np.linalg.eigh(Hhat)
    Stilde = Q.T @ S @ Q
    Ptilde = Q.T @ P @ Q
    lam_k = np.empty(d)
    for k in range(d):
        denom = Ptilde[k, k]
        if denom <= 1e-30 or Stilde[k, k] <= 0:
            lam_k[k] = 0.5 * (2.0 / mu[k]) * 0.5
        else:
            lk = 2.0 * mu[k] * Stilde[k, k] / denom
            lam_k[k] = min(lk, 0.95 * 2.0 / mu[k])
    Lam0 = Q @ np.diag(lam_k) @ Q.T
    Lam0 = 0.5 * (Lam0 + Lam0.T)
    starts = [Lam0[np.triu_indices(d)] if symmetric else Lam0.flatten()]
    # a few scaled variants of the eigenbasis start
    for frac in (0.5, 0.75, 1.0, 1.1):
        Ls = frac * Lam0
        if np.all(np.abs(np.linalg.eigvals(np.eye(d) - Ls @ Hhat)) < 1 - 1e-9):
            starts.append(Ls[np.triu_indices(d)] if symmetric else Ls.flatten())

    best = None
    for x0 in starts:
        try:
            sol = least_squares(F, x0, method="lm", xtol=1e-14, ftol=1e-14, max_nfev=6000)
            cost = float(sol.cost)
        except Exception:
            cost = np.inf
            sol = None
        if sol is None:
            continue
        Lam = make_Lam(sol.x)
        if nmap is not None:
            try:
                Sig_act = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement, nmap=nmap)[0]
                rel = float(np.linalg.norm(Sig_act - S) / max(np.linalg.norm(S), 1e-30))
            except Exception:
                Sig_act = np.full_like(S, np.nan)
                rel = float("inf")
        else:
            Sig_act = np.full_like(S, np.nan)
            rel = float("nan")
        key = rel if np.isfinite(rel) else 1e6
        if best is None or key < best[2]:
            best = (Lam, Sig_act, rel)
        if rel < 1e-5:
            break
    if best is None:
        best = (np.full((d, d), np.nan), np.full_like(S, np.nan), float("inf"))
    return best


def solve_full_lambda_for_target(target_Sig, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta,
                                 replacement=True, x0=None):
    """Algorithm 1 step 1 (DQ+exact): Cbar_fixed is the EXACT Eq16 noise at the target."""
    nmap = _noise_linear_map(Jn, gn, N, B, replacement)
    Cbar_fixed = noise_cov_eq16(Jn, gn, 0.5 * (target_Sig + target_Sig.T), Gamma, N, B, replacement)
    return _solve_lambda_eq15(target_Sig, Hhat, Cbar_fixed, beta, nmap=nmap, Jn=Jn, gn=gn,
                              Gamma=Gamma, theta_hat=theta_hat, N=N, B=B, replacement=replacement)


def _solve_full_lambda_generic(target_Sig, Hhat, predict_fn, x0=None):
    """Find full-matrix Lambda s.t. predict_fn(Lambda) == target_Sig (Powell hybrid)."""
    from scipy.optimize import root
    d = Hhat.shape[0]
    target_sym = 0.5 * (target_Sig + target_Sig.T)
    lam_max = 2.0 / np.linalg.eigvalsh(Hhat).max() * 0.999

    def F(vec):
        Lam = vec.reshape(d, d)
        ev = np.linalg.eigvals(np.eye(d) - Lam @ Hhat)
        if np.any(np.abs(ev) >= 1.0 - 1e-9):
            return np.ones(d * d) * 1e3
        try:
            Sig = predict_fn(Lam)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.ones(d * d) * 1e3
        if not np.all(np.isfinite(Sig)):
            return np.ones(d * d) * 1e3
        return (Sig - target_sym).flatten()

    if x0 is None:
        x0 = (0.5 * lam_max * np.eye(d)).flatten()
    sol = root(F, x0, method="hybr", options={"maxfev": 3000 * d * d, "xtol": 1e-11})
    Lam = sol.x.reshape(d, d)
    try:
        Sig = predict_fn(Lam)
        rel = float(np.linalg.norm(Sig - target_sym) / max(np.linalg.norm(target_sym), 1e-30))
    except Exception:
        Sig = np.full_like(target_sym, np.nan)
        rel = float("inf")
    return Lam, Sig, rel


def method_covariance_error(method, target_Sig, Sstar, Hhat, Jn, gn, Gamma, theta_hat,
                            N, B, beta, A_ws=None, sigma2_ws=None, replacement=True):
    """Algorithm-1 comparison: each method chooses Lambda to match target_Sig under ITS noise
    model; the *actual* stationary covariance is then computed under the EXACT noise (Eq15+16)
    and compared to the true sandwich S*. Returns dict with lambda, rel_err_to_target,
    cov_error_to_Sstar (the Table-2/3 metric), and actual Sigma. 'inf' cov_error = divergence.
    """
    d = Hhat.shape[0]
    nmap = _noise_linear_map(Jn, gn, N, B, replacement)
    S = 0.5 * (target_Sig + target_Sig.T)
    common = dict(Jn=Jn, gn=gn, Gamma=Gamma, theta_hat=theta_hat, N=N, B=B,
                  replacement=replacement, nmap=nmap)
    Sig_pred = np.full_like(S, np.nan)
    rel = float("nan")
    if method == "DQ+exact":
        # exact Eq16 noise evaluated at the target
        Cbar_fixed = noise_cov_eq16(Jn, gn, S, Gamma, N, B, replacement)
        Lam, Sig_pred, rel = solve_lambda_eigenbasis(S, Hhat, Cbar_fixed, beta, **common)
    elif method == "DQ+const":
        # constant-noise heuristic: Cbar = Hhat / B (Eq 9, Liu/Dieuleveut)
        Lam, Sig_pred, rel = solve_lambda_eigenbasis(S, Hhat, Hhat / B, beta, **common)
    elif method == "LR+WS":
        # well-specified approximation Eq 10: Cbar = B^-1 (A S A + tr[AS] A + sigma^2 A)
        if A_ws is None or sigma2_ws is None:
            raise ValueError("LR+WS needs A_ws (covariate 2nd moment) and sigma2_ws")
        Aw, s2 = A_ws, sigma2_ws
        Cbar_fixed = (Aw @ S @ Aw + np.trace(Aw @ S) * Aw + s2 * Aw) / B
        Lam, Sig_pred, rel = solve_lambda_eigenbasis(S, Hhat, Cbar_fixed, beta, **common)
    elif method == "CT":
        # continuous-time theory: stationary covariance is the OU-process limit, independent
        # of the step size, solving  Hhat Sigma + Sigma Hhat = Cchat  (Eq 6/7). The discrete
        # chain's actual covariance (sigma_psi_exact under some Lambda) is not defined by CT,
        # so we report the CT-predicted covariance itself vs S* (the paper's CT column).
        from scipy.linalg import solve_continuous_lyapunov as scl
        gbar = gn.mean(axis=0)
        Cchat = (np.einsum("ni,nj->ij", gn, gn) / N - np.outer(gbar, gbar)) / B
        try:
            Sig_pred = scl(Hhat, Cchat)
            Sig_pred = 0.5 * (Sig_pred + Sig_pred.T)
            if not np.all(np.isfinite(Sig_pred)) or np.any(np.linalg.eigvalsh(Sig_pred) < 0):
                raise ValueError
            rel = float(np.linalg.norm(Sig_pred - S) / max(np.linalg.norm(S), 1e-30))
            Lam = np.full((d, d), np.nan)
        except Exception:
            Sig_pred = np.full_like(S, np.nan)
            Lam = np.full((d, d), np.nan)
            rel = float("inf")
    else:
        raise ValueError(method)
    # CT has no discrete Lambda; its covariance IS the OU-predicted one. For the discrete
    # methods, the *actual* chain covariance is computed under the EXACT Eq16 noise.
    if method == "CT":
        Sig_actual = Sig_pred
    else:
        try:
            Sig_actual = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement, nmap=nmap)[0]
        except Exception:
            Sig_actual = np.full_like(Sstar, np.nan)
    if (not np.all(np.isfinite(Sig_actual))) or (np.linalg.norm(Sig_actual) > 1e14):
        cov_err = float("inf")
    elif np.any(np.linalg.eigvalsh(Sig_actual) < -1e-9 * max(1.0, np.max(np.abs(Sig_actual)))):
        cov_err = float("inf")
    else:
        cov_err = float(np.linalg.norm(Sig_actual - Sstar) / max(np.linalg.norm(Sstar), 1e-30))
    return dict(method=method, Lambda=Lam, rel_err_to_target=float(rel),
                cov_error_to_Sstar=cov_err, Sigma_actual=Sig_actual)


def actual_sigma_with_lambda(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement=True):
    """The *actual* discrete-time stationary covariance under the EXACT noise model (Eq15+16),
    for a given (possibly method-chosen) Lambda. This is what each method's chain really
    looks like; the covariance error compares this to the true sandwich S*."""
    Sig, _ = sigma_psi_exact(Lam, Hhat, Jn, gn, Gamma, theta_hat, N, B, beta, replacement)
    return Sig


def solve_lambda_lrws(target_Sig, Hhat, A, sigma2, B, beta):
    """LR+WS tuning (Ziyin 2022, Eq 9+10): assumes well-specified z~N(0,A) so
    Cbar_psi = B^-1 (A Sigma A + tr[A Sigma] A + sigma^2 A). Solve for lambda.
    Under strong misspecification this produces an unstable lambda -> large error."""
    from scipy.optimize import minimize_scalar
    d = Hhat.shape[0]
    lam_max = 2.0 / np.linalg.eigvalsh(Hhat).max() * 0.999

    def relerr(lam):
        if lam <= 0:
            return 1e9
        Lam = lam * np.eye(d)
        # iterate Eq9 with the WS-noise model (Eq 10)
        A_mat = np.eye(d) - Lam @ Hhat
        inj = 0.0 if not np.isfinite(beta) else 2.0 / beta
        Sig = np.eye(d) * 1e-6
        for _ in range(300):
            Cbar = (A @ Sig @ A + np.trace(A @ Sig) * A + sigma2 * A) / B
            Q = Lam @ Cbar @ Lam + inj * Lam
            Sn = solve_discrete_lyapunov(A_mat, Q)
            if np.max(np.abs(Sn - Sig)) < 1e-12 * max(1.0, np.max(np.abs(Sig))):
                Sig = Sn
                break
            Sig = Sn
        if not np.all(np.isfinite(Sig)) or np.any(np.linalg.eigvalsh(Sig) < 0):
            return 1e9
        return float(np.linalg.norm(Sig - target_Sig) / max(np.linalg.norm(target_Sig), 1e-30))

    res = minimize_scalar(relerr, bounds=(1e-10, lam_max), method="bounded",
                          options={"xatol": 1e-12})
    lam = res.x
    return lam, None, float(res.fun)


# --------------------------------------------------------------------------- #
# Momentum (Proposition B.1, Eq B.5)                                           #
# --------------------------------------------------------------------------- #
def sigma_psi_momentum(Lam, Hhat, Cbar, kappa, beta):
    """Stationary covariance of SGLD-with-momentum proxy (Prop B.1).

    Augmented state s_t=[delta_psi_t; nu_t] (delta_psi = psi - theta_hat) is linear-Gaussian:
        nu_t   = kappa nu_{t-1} + Hhat delta_psi_{t-1} + eta_t        (eta = grad noise, cov Cbar)
        delta_psi_t = (I - Lam Hhat) delta_psi_{t-1} - Lam kappa nu_{t-1}
                      - Lam eta_t + sqrt(2/beta Lam) xi_t
    One-step matrix A and noise covariance Q built accordingly; the stationary
    covariance is the (0,0) block of solve_discrete_lyapunov(A, Q). This is the exact
    equivalent of Eq B.5, and kappa=0 recovers sigma_psi (Eq 15)."""
    d = Hhat.shape[0]
    I = np.eye(d)
    A11 = I - Lam @ Hhat
    A12 = -Lam * kappa
    A21 = Hhat
    A22 = kappa * I
    A = np.block([[A11, A12], [A21, A22]])
    inj = 0.0 if not np.isfinite(beta) else 2.0 / beta
    # noise w_t = [ -Lam eta_t + sqrt(2/beta Lam) xi_t ; eta_t ]
    Q = np.zeros((2 * d, 2 * d))
    Q[:d, :d] = Lam @ Cbar @ Lam + inj * Lam          # var of (-Lam eta + injected)
    Q[:d, d:] = -Lam @ Cbar                            # cross  (-Lam eta) with eta
    Q[d:, :d] = -Cbar @ Lam
    Q[d:, d:] = Cbar                                   # var of eta
    S = solve_discrete_lyapunov(A, Q)
    return S[:d, :d]


# --------------------------------------------------------------------------- #
# Monte-Carlo validators                                                       #
# --------------------------------------------------------------------------- #
def mc_noise_cov(Jn, gn, Sigma, theta_hat, N, B, n_trials=4000, seed=0, replacement=True):
    """Empirical minibatch-gradient noise covariance: sample minibatches at stationarity
    (psi-theta_hat ~ N(0, Sigma)) and average the gradient-noise outer products.
    Must match noise_cov_eq16 to MC error."""
    rng = np.random.default_rng(seed)
    d = gn.shape[1]
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))
    gbar = gn.mean(axis=0)
    Jbar = Jn.mean(axis=0)
    acc = np.zeros((d, d))
    for _ in range(n_trials):
        delta = L @ rng.standard_normal(d)          # psi - theta_hat ~ N(0, Sigma)
        # per-sample stochastic gradient of the quadratic proxy:
        u = gn + np.einsum("nij,j->ni", Jn, delta)   # u_n = g_n + J_n delta
        if replacement:
            idx = rng.integers(0, N, size=B)
        else:
            idx = rng.choice(N, size=B, replace=False)
        ub = u[idx].mean(axis=0)                     # minibatch mean
        ubar = u.mean(axis=0)                         # full mean (drift)
        eta = ub - ubar
        acc += np.outer(eta, eta)
    return acc / n_trials


def sgd_chain_mb(grad_mb, theta_hat, lam, B, N, n_steps, burn, thin, seed=0,
                 beta=np.inf, d=None):
    """Fast constant-step SGD/SGLD chain. grad_mb(theta, idx) returns the (d,) mean gradient over
    the minibatch `idx` (length B) -- so only B per-sample gradients are computed per step (not N).
    Returns stationary samples (n_steps, d)."""
    rng = np.random.default_rng(seed)
    d = theta_hat.shape[0] if d is None else d
    x = theta_hat.copy()
    inj = 0.0 if not np.isfinite(beta) else np.sqrt(2.0 / beta * lam)

    def step():
        idx = rng.integers(0, N, size=B)
        g = grad_mb(x, idx)
        x2 = x - lam * g
        if inj:
            x2 = x2 + inj * rng.standard_normal(d)
        return x2

    for _ in range(burn):
        x = step()
    for _ in range(thin):
        x = step()
    samples = np.empty((n_steps, d))
    samples[0] = x
    for k in range(1, n_steps):
        for _ in range(thin):
            x = step()
        samples[k] = x
    return samples


def sgd_chain_true(grad_per_sample, theta_hat, lam, B, N, n_steps, burn, thin, seed=0,
                   beta=np.inf, reg=0.0):
    """Run constant-step SGD/SGLD on the *true* (non-quadratic) per-sample loss and collect
    stationary samples. grad_per_sample(theta) -> (N, d) array of per-sample gradients.
    Returns (samples, ) where samples is (n_collect, d). (Slower than sgd_chain_mb.)"""
    rng = np.random.default_rng(seed)
    d = theta_hat.shape[0]
    x = theta_hat.copy()
    inj = 0.0 if not np.isfinite(beta) else np.sqrt(2.0 / beta * lam)
    for _ in range(burn):
        G = grad_per_sample(x)                       # (N, d)
        idx = rng.integers(0, N, size=B)
        g = G[idx].mean(axis=0) + reg * x
        x = x - lam * g
        if inj:
            x = x + inj * rng.standard_normal(d)
    samples = np.empty((n_steps, d))
    for k in range(n_steps):
        for _ in range(thin):
            G = grad_per_sample(x)
            idx = rng.integers(0, N, size=B)
            g = G[idx].mean(axis=0) + reg * x
            x = x - lam * g
            if inj:
                x = x + inj * rng.standard_normal(d)
        samples[k] = x
    return samples


# --------------------------------------------------------------------------- #
# Wasserstein (Gaussian / Bures) and regression helpers                        #
# --------------------------------------------------------------------------- #
def w2_gaussian(mu1, S1, mu2, S2):
    """Exact 2-Wasserstein distance between two Gaussians N(mu1,S1), N(mu2,S2)."""
    S1h = sqrtm(S1)
    inner = S1h @ S2 @ S1h
    ev = np.maximum(np.real(np.linalg.eigvalsh(inner)), 0.0)
    return float(np.sqrt(np.sum((mu1 - mu2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * np.sum(np.sqrt(ev))))


def fit_loglog(xs, ys):
    """Fit log(ys) = a + p log(xs); return (p, a, resid)."""
    lx, ly = np.log(xs), np.log(ys)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (p, a), *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = a + p * lx
    resid = float(np.sqrt(np.mean((pred - ly) ** 2)))
    return float(p), float(a), resid


def logistic_data(N, D, theta_star=None, seed=0, reg_for_strong_convexity=1e-3):
    """Synthetic logistic-regression dataset satisfying Assumptions (A)-(C)."""
    rng = np.random.default_rng(seed)
    if theta_star is None:
        theta_star = rng.standard_normal(D) * 0.5
    Z = rng.standard_normal((N, D))
    eta = Z @ theta_star
    p = 1.0 / (1.0 + np.exp(-eta))
    y = np.where(rng.random(N) < p, 1.0, -1.0)
    return Z, y, theta_star


def logistic_mle(Z, y, reg=1e-3, max_iter=200, tol=1e-12):
    """MLE for logistic regression with L2 reg (Newton)."""
    N, D = Z.shape
    th = np.zeros(D)
    for _ in range(max_iter):
        eta = Z @ th
        s = 1.0 / (1.0 + np.exp(-eta))
        g = -Z.T @ ((y + 1) / 2 - s) / N + reg * th
        W = s * (1 - s)
        H = (Z * W[:, None]).T @ Z / N + reg * np.eye(D)
        step = np.linalg.solve(H, g)
        th = th - step
        if np.linalg.norm(step) < tol:
            break
    return th


def logistic_grads_hessians(Z, y, theta_hat, reg=1e-3):
    """Per-sample gradients g_n and Hessians J_n for logistic loss at theta_hat."""
    N, D = Z.shape
    eta = Z @ theta_hat
    s = 1.0 / (1.0 + np.exp(-eta))
    py = (y + 1) / 2.0
    g = (s - py)[:, None] * Z + reg * theta_hat[None, :]      # (N,D) per-sample grad incl reg
    w = s * (1 - s)                                          # (N,)
    Jn = np.empty((N, D, D))
    reg_eye = reg * np.eye(D)
    for n in range(N):
        Jn[n] = w[n] * np.outer(Z[n], Z[n]) + reg_eye / N     # reg term averaged per sample
    return g, Jn


def linear_regression_sandwich(Z, y, sigma2=1.0):
    """For linear regression ell_n = (y_n - theta' z_n)^2 / (2 sigma^2):
    theta_hat (OLS), per-sample grads, Hessians, and the sandwich target S* = J^-1 I J^-1.
    sigma2 defaults to 1 (the sandwich S* is invariant to sigma2; using 1 gives the best
    numerical conditioning for the Lambda root-finder)."""
    N, D = Z.shape
    theta_hat = np.linalg.lstsq(Z, y, rcond=None)[0]
    r = y - Z @ theta_hat
    gn = -(r / sigma2)[:, None] * Z                            # (N,D) grad of ell_n at hat
    Jn = np.empty((N, D, D))
    for n in range(N):
        Jn[n] = np.outer(Z[n], Z[n]) / sigma2
    Jbar = Jn.mean(axis=0)
    Ical = gn.T @ gn / N
    Sstar = np.linalg.solve(Jbar, Ical) @ np.linalg.inv(Jbar)
    return theta_hat, gn, Jn, Jbar, Ical, Sstar, sigma2
