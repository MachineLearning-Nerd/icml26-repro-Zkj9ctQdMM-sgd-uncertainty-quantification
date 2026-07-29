"""Claim 5 falsification campaign (Table 3, Boston housing).

Registered claim: "On the Boston housing dataset under strong model misspecification, the
proposed DQ+exact method maintains accurate covariance estimates while competing
continuous-time and constant-noise methods diverge (Table 3)."

The paper's own Table 3 (log loss, covariance error) reports the continuous-time tuning
CT = 0.247 (B=16) and 0.589 (B=|0.1N|) -- FINITE, with tight 95% CIs -- while LR+WS =
9.23e8 / 1.40e7 (divergent). The registered claim quantifies divergence over BOTH the
continuous-time and constant-noise method families; the CT conjunct is the falsification
target: Table 3 itself never shows CT diverging.

This module independently replays the full 506x13 Boston Table-3 setting with NO
stability clamps on any tuning (a clamped solver was the flaw of the earlier toy-scale
attempt: it forces every method to be finite by construction, so it can neither
reproduce nor refute divergence).

Model contract (paper Sec 3/6.1): linear model, log loss l_n = (1/2)(y_n - theta'z_n)^2,
i.e. the assumed observation model is y | z ~ N(theta'z, sigma^2=1). On raw-scale Boston
medv the true residual variance is ~21.9 -- the strong misspecification the claim
conditions on. LR+WS ("linear regression + well-specified", eq (6)) trusts the model
noise sigma^2 = 1; that ~22x noise underestimate is the divergence mechanism.

Two independent evidence routes per (method, batch size):
  1. EXACT: for linear regression the SGD chain is an exactly linear random recursion
     theta_t - theta_hat = (I - Lam H_B)(theta_{t-1} - theta_hat) - Lam g_B(theta_hat),
     so the second moment evolves under the closed-form linear operator
        T = E[(I - Lam H_B) kron (I - Lam H_B)]   (D^2 x D^2)
     -- no diffusion approximation, no Monte Carlo. The chain diverges iff rho(T) >= 1;
     when rho(T) < 1 the exact stationary covariance solves
        (I - T) vec(S) = (Lam kron Lam) vec(C0).
  2. SIMULATION: 30 independent real-minibatch SGD chains per cell (the paper's
     30-repetition protocol), empirical covariance error to the sandwich target.

Tunings (Algorithm 1, step 1), target Sigma_psi = S*/N (sandwich at posterior scale):
  DQ+exact : per-mode solve of eq (11) with the exact eq-(12) noise at the target.
  DQ+const : same solve with the constant-noise heuristic C = Hhat/B (paper Sec E.2).
  LR+WS    : same solve with the well-specified eq-(6) noise, model sigma^2 = 1.
  CT       : eq (3) applied in the eigenbasis of Hhat (the form whose finite errors
             match the paper's own CT column).
  Sensitivity: CT-asym (literal asymmetric eq (3)), LR+WS-sighat (estimated residual
  variance instead of model sigma^2), DQ+exact-asym (closed-form exact solution
  Lam = 2 S H (C + H S H)^{-1} of eq (11) -- must reproduce the target exactly,
  an internal cross-check of the whole machinery).

Controls:
  * positive control -- LR+WS must diverge at both batch sizes (the claim's own
    asserted phenomenon; places the replay inside the claim's regime);
  * negative control (detector) -- CT's Lambda scaled x50 must be flagged divergent by
    BOTH routes (CT's finiteness is not a detection failure);
  * negative control (assumption) -- with Gaussian covariates (same second moment) and
    well-specified unit-variance responses, LR+WS's assumptions hold exactly and it
    must be stable AND accurate: the Boston divergence is caused by the
    misspecification the claim conditions on, not by our implementation.

Protocol degrees of freedom the paper leaves open (y preprocessing, target scale) are
swept. PRE-REGISTERED primary-variant rule: a variant qualifies iff BOTH stated premises
of the claim's regime reproduce -- DQ+exact accurate (rho<1, err<1 at both B) AND LR+WS
divergent at both B; ties broken by CT error closest to the paper's 0.247 (B=16). If no
variant qualifies the campaign is BLOCKED (no valid falsification). CT finiteness is
additionally reported across ALL posterior-scale variants and the without-replacement
sensitivity.

Outputs: outputs/c5_falsification.json. Deterministic verifier:
repro/src/check_claim5_falsification.py (exits nonzero when the falsification is
absent). Run: uv run --locked python repro/src/claim5_falsify.py [--fast]
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time

import numpy as np

SEED = 20260729
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "boston.csv")

PAPER_TABLE3_LOG = {  # log-loss covariance-error row of Table 3 (exact transcription)
    "16": {"Posterior": 0.358, "CT": 0.247, "LR+WS": 9.23e8, "DQ+exact": 0.337},
    "50": {"Posterior": 0.358, "CT": 0.589, "LR+WS": 1.40e7, "DQ+exact": 0.352},
}
PAPER_CT_CI = {"16": [0.194, 0.310], "50": [0.443, 0.804]}   # Table D.4 (all finite)
DIVERGE_THRESH = 1.0e3   # >= this covariance error counts as "diverged" (paper's
                         # divergent entries are >= 1.40e7; its finite entries <= 3.2)
FINITE_THRESH = 10.0     # a "does not diverge" cell must stay below this

PRIMARY_METHODS = ("DQ+exact", "DQ+const", "LR+WS", "CT")
SENSITIVITY_METHODS = ("CT-asym", "LR+WS-sighat", "DQ+exact-asym")


# --------------------------------------------------------------------------- #
# Data and sandwich                                                            #
# --------------------------------------------------------------------------- #
def load_boston(preproc):
    """preproc: 'icpt' (raw y + intercept col), 'ceny' (centered y), 'stdy' (standardized y)."""
    rows = list(csv.DictReader(open(DATA)))
    feat = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax",
            "ptratio", "b", "lstat"]
    Z = np.array([[float(r[f]) for f in feat] for r in rows], dtype=float)
    y = np.array([float(r["medv"]) for r in rows], dtype=float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    if preproc == "icpt":
        Z = np.hstack([np.ones((Z.shape[0], 1)), Z])
    elif preproc == "ceny":
        y = y - y.mean()
    elif preproc == "stdy":
        y = (y - y.mean()) / y.std()
    else:
        raise ValueError(preproc)
    return Z, y


def sandwich(Z, y):
    """OLS theta_hat; per-sample grads; J = (1/N)Z'Z; I = (1/N) sum g g'; S* = J^-1 I J^-1."""
    N, D = Z.shape
    theta_hat = np.linalg.lstsq(Z, y, rcond=None)[0]
    r = y - Z @ theta_hat
    gn = -r[:, None] * Z                              # per-sample gradient at theta_hat
    Jbar = Z.T @ Z / N
    Ical = gn.T @ gn / N
    Ji = np.linalg.inv(Jbar)
    Sstar = Ji @ Ical @ Ji
    resid_var = float(np.mean(r ** 2))
    return theta_hat, gn, Jbar, Ical, Sstar, resid_var


# --------------------------------------------------------------------------- #
# Exact second-moment operator (route 1)                                       #
# --------------------------------------------------------------------------- #
def hessian_fourth_moment(Z, B, replacement=True):
    """M with (M vec(S))_ij = E[(H_B S H_B)]_ij - (Jbar S Jbar)_ij for a size-B minibatch
    of iid uniform draws (H_B = (1/B) sum z z'). With replacement:
    E[H_B S H_B] - Jbar S Jbar = (1/B)(avg_n J_n S J_n - Jbar S Jbar)."""
    N, D = Z.shape
    Jbar = Z.T @ Z / N
    ZZ = np.einsum("ni,nj->nij", Z, Z).reshape(N, D * D)
    M4 = (ZZ.T @ ZZ) / N                              # E[vec(zz') vec(zz')'] : [ik],[jl]
    M4 = M4.reshape(D, D, D, D)                       # [i,k,j,l] = E z_i z_k z_j z_l
    A4 = np.transpose(M4, (0, 2, 1, 3)).reshape(D * D, D * D)   # [(i,j),(k,l)]
    Jterm = np.einsum("ik,jl->ijkl", Jbar, Jbar).reshape(D * D, D * D)
    M = (A4 - Jterm) / B
    if not replacement:
        M *= (N - B) / (N - 1)
    return M, Jbar


def hessian_fourth_moment_apply(Z, S):
    """avg_n J_n S J_n with J_n = z_n z_n' :  avg_n (z' S z) z z'."""
    q = np.einsum("ni,ij,nj->n", Z, S, Z)
    return np.einsum("n,ni,nj->ij", q, Z, Z) / Z.shape[0]


def second_moment_operator(Lam, Z, B, replacement=True, mmap=None):
    """T = kron(A, A) + kron(Lam, Lam) @ M with A = I - Lam Jbar. rho(T) >= 1 <=> the real
    chain's second moment diverges (exact for the quadratic log loss -- no approximation)."""
    if mmap is None:
        mmap = hessian_fourth_moment(Z, B, replacement)
    M, Jbar = mmap
    D = Jbar.shape[0]
    A = np.eye(D) - Lam @ Jbar
    T = np.kron(A, A) + np.kron(Lam, Lam) @ M
    rho = float(np.max(np.abs(np.linalg.eigvals(T))))
    return T, rho


def exact_stationary_cov(Lam, Z, gn, B, replacement=True, mmap=None):
    """Exact stationary covariance of the real Boston SGD chain under Lam.
    Returns (Sigma, rho); Sigma = None when rho >= 1 (divergent)."""
    N, D = Z.shape
    T, rho = second_moment_operator(Lam, Z, B, replacement, mmap)
    if rho >= 1.0:
        return None, rho
    C0 = (gn.T @ gn / N - np.outer(gn.mean(0), gn.mean(0))) / B
    if not replacement:
        C0 *= (N - B) / (N - 1)
    rhs = np.kron(Lam, Lam) @ C0.reshape(-1)
    S = np.linalg.solve(np.eye(D * D) - T, rhs).reshape(D, D)
    return 0.5 * (S + S.T), rho


# --------------------------------------------------------------------------- #
# Tunings (Algorithm 1, step 1) -- NO stability clamps                         #
# --------------------------------------------------------------------------- #
def tune_all(St, Jbar, Ical, Z, B, resid_var):
    """Solve each method's tuning equation for Lambda with target Sigma_psi = St.

    Eigenbasis per-mode solve of eq (5)/(11) (Hhat = Q diag(h) Q'):
        lam_k = 2 h_k S~_kk / (C~_kk + h_k^2 S~_kk)     [UNCLAMPED]
    CT eq (3): Lam = (S H + H S) Chat^{-1}; eigenbasis form drops the h^2 S~ term.
    Closed-form exact solution of eq (5): Lam = 2 S H (C + H S H)^{-1} (verified by
    substitution: Lam H S + S H Lam' = Lam (C + HSH) Lam')."""
    D = Jbar.shape[0]
    h, Q = np.linalg.eigh(Jbar)
    Stt = Q.T @ St @ Q
    Chat = Ical / B                                  # gradient noise at theta_hat
    HSH = Jbar @ St @ Jbar

    def diag_solve(Cmat, drop_hsh=False):
        Ct = Q.T @ Cmat @ Q
        denom = np.diag(Ct) + (0.0 if drop_hsh else h ** 2 * np.diag(Stt))
        lam = 2.0 * h * np.diag(Stt) / denom
        return Q @ np.diag(lam) @ Q.T

    JSJ = hessian_fourth_moment_apply(Z, St)
    C_exact = (Ical + JSJ - HSH) / B                 # eq (12) noise at the target
    C_const = Jbar / B                               # constant-noise heuristic
    C_ws1 = (HSH + np.trace(Jbar @ St) * Jbar + 1.0 * Jbar) / B        # eq (6), model s2=1
    C_wsh = (HSH + np.trace(Jbar @ St) * Jbar + resid_var * Jbar) / B  # eq (6), est. s2

    lams = {
        "DQ+exact": diag_solve(C_exact),
        "DQ+const": diag_solve(C_const),
        "LR+WS": diag_solve(C_ws1),
        "CT": diag_solve(Chat, drop_hsh=True),
        "CT-asym": (St @ Jbar + Jbar @ St) @ np.linalg.inv(Chat),
        "LR+WS-sighat": diag_solve(C_wsh),
        "DQ+exact-asym": 2.0 * St @ Jbar @ np.linalg.inv(C_exact + HSH),
    }
    return lams


# --------------------------------------------------------------------------- #
# Real SGD simulation (route 2)                                                #
# --------------------------------------------------------------------------- #
def simulate(Lam, Z, y, theta_hat, St, B, reps, burn, iters, seed, replacement=True):
    """`reps` vectorized real-minibatch SGD chains; per-rep covariance error to St."""
    rng = np.random.default_rng(seed)
    N, D = Z.shape
    X = np.tile(theta_hat, (reps, 1))
    alive = np.ones(reps, dtype=bool)
    blow = 1e9 * max(np.sqrt(abs(np.trace(St))), 1e-12) + 1e6 * (1 + np.linalg.norm(theta_hat))
    s1 = np.zeros((reps, D))
    s2 = np.zeros((reps, D, D))
    cnt = 0
    LamT = np.ascontiguousarray(Lam.T)
    total = burn + iters
    CH = 512                                          # draw minibatch indices in chunks
    t = 0
    while t < total:
        n = min(CH, total - t)
        if replacement:
            idx = rng.integers(0, N, size=(n, reps, B))
        else:
            idx = np.stack([np.stack([rng.choice(N, size=B, replace=False)
                                      for _ in range(reps)]) for _ in range(n)])
        for s in range(n):
            Zb = Z[idx[s]]                                        # (R, B, D)
            resid = np.einsum("rbd,rd->rb", Zb, X) - y[idx[s]]
            g = np.einsum("rb,rbd->rd", resid, Zb) / B
            X = X - g @ LamT
            bad = ~np.isfinite(X).all(axis=1) | (np.abs(X).max(axis=1) > blow)
            if bad.any():
                alive &= ~bad
                X[bad] = 0.0
            if t + s >= burn:
                s1 += np.where(alive[:, None], X, 0.0)
                s2 += np.where(alive[:, None, None], X[:, :, None] * X[:, None, :], 0.0)
                cnt += 1
        t += n
    errs = np.full(reps, np.inf)
    nrm = np.linalg.norm(St)
    for r_ in range(reps):
        if alive[r_]:
            mu = s1[r_] / cnt
            Sig = s2[r_] / cnt - np.outer(mu, mu)
            errs[r_] = np.linalg.norm(Sig - St) / nrm
    fin = errs[np.isfinite(errs)]
    out = dict(finite_runs=int(alive.sum()), diverged_runs=int(reps - alive.sum()),
               per_rep=[float(e) for e in errs])
    if len(fin):
        half = 1.96 * fin.std(ddof=1) / np.sqrt(len(fin)) if len(fin) > 1 else 0.0
        out.update(mean=float(fin.mean()), median=float(np.median(fin)),
                   min=float(fin.min()), max=float(fin.max()),
                   ci95=[float(fin.mean() - half), float(fin.mean() + half)])
    else:
        out.update(mean=float("inf"), median=float("inf"), min=float("inf"),
                   max=float("inf"), ci95=[float("inf"), float("inf")])
    return out


# --------------------------------------------------------------------------- #
# One cell / one variant                                                       #
# --------------------------------------------------------------------------- #
def eval_method(name, Lam, Z, y, theta_hat, gn, St, B, mmap, reps, burn, iters, seed,
                replacement=True, simulate_it=True):
    Sig_exact, rho = exact_stationary_cov(Lam, Z, gn, B, replacement, mmap)
    nrm = np.linalg.norm(St)
    exact_err = float("inf") if Sig_exact is None else float(np.linalg.norm(Sig_exact - St) / nrm)
    ev = np.real(np.linalg.eigvals(Lam @ (Z.T @ Z / Z.shape[0])))
    tau = float(2.0 / max(np.min(ev), 1e-300)) if np.min(ev) > 0 else float("inf")
    cell = dict(method=name, rho=float(rho), exact_cov_err=exact_err,
                analytic_diverges=bool(rho >= 1.0), tau_mix_iters=tau)
    if simulate_it:
        sim = simulate(Lam, Z, y, theta_hat, St, B, reps, burn, iters, seed, replacement)
        cell["sim"] = sim
        cell["diverged"] = bool(rho >= 1.0 or sim["diverged_runs"] > 0
                                or sim["median"] >= DIVERGE_THRESH)
    else:
        cell["diverged"] = bool(rho >= 1.0)
    return cell


def run_variant(tag, preproc, nscale, B_list, reps, burn, iters, simulate_it=True,
                replacement=True, control=None, seed0=SEED):
    """control: None | 'wellspec-gauss' (Gaussian z, well-specified unit-noise y)."""
    Z, y = load_boston(preproc)
    N, D = Z.shape
    if control == "wellspec-gauss":
        th0 = np.linalg.lstsq(Z, y, rcond=None)[0]
        Jb0 = Z.T @ Z / N
        rng = np.random.default_rng(seed0 + 777)
        Z = rng.standard_normal((N, D)) @ np.linalg.cholesky(Jb0 + 1e-12 * np.eye(D)).T
        y = Z @ th0 + rng.standard_normal(N)          # model N(theta'z, 1) is CORRECT here
    theta_hat, gn, Jbar, Ical, Sstar, resid_var = sandwich(Z, y)
    St = Sstar / N if nscale else Sstar
    res = dict(tag=tag, preproc=preproc, nscale=nscale, N=int(N), D=int(D),
               control=control, resid_var=resid_var, replacement=replacement,
               target_scale=("Sstar/N" if nscale else "Sstar"), cells={})
    post = resid_var * np.linalg.inv(Jbar) / N        # homoskedastic-posterior reference
    res["posterior_cov_err"] = float(np.linalg.norm(post - Sstar / N) / np.linalg.norm(Sstar / N))
    for B in B_list:
        mmap = hessian_fourth_moment(Z, B, replacement)
        lams = tune_all(St, Jbar, Ical, Z, B, resid_var)
        row = {}
        for i, m in enumerate(PRIMARY_METHODS + SENSITIVITY_METHODS):
            row[m] = eval_method(m, lams[m], Z, y, theta_hat, gn, St, B, mmap,
                                 reps, burn, iters, seed0 + 1000 * B + i, replacement,
                                 simulate_it)
            evs = np.real(np.linalg.eigvals(lams[m]))
            row[m]["lam_minmax"] = [float(evs.min()), float(evs.max())]
        if simulate_it and control is None:
            row["CTx50-control"] = eval_method(
                "CTx50-control", 50.0 * lams["CT"], Z, y, theta_hat, gn, St, B, mmap,
                min(reps, 8), min(burn, 2000), min(iters, 4000),
                seed0 + 1000 * B + 99, replacement, True)
        res["cells"][str(B)] = row
    return res


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "inf"
    return f"{x:.3g}"


def print_variant(v):
    print(f"\n--- variant {v['tag']}: preproc={v['preproc']} target={v['target_scale']}"
          f" control={v['control']} repl={v['replacement']}"
          f" (N={v['N']} D={v['D']} resid_var={v['resid_var']:.3f})", flush=True)
    for B, row in v["cells"].items():
        for m, c in row.items():
            sim = c.get("sim")
            simtxt = (f" sim:{sim['finite_runs']}fin/{sim['diverged_runs']}div"
                      f" med={fmt(sim['median'])}") if sim else ""
            print(f"    B={B:>2} {m:<14} rho={c['rho']:.4f} exact_err={fmt(c['exact_cov_err'])}"
                  f"{simtxt} -> {'DIVERGES' if c['diverged'] else 'finite'}", flush=True)


# --------------------------------------------------------------------------- #
# Campaign driver                                                              #
# --------------------------------------------------------------------------- #
def run_all(fast=False):
    t0 = time.time()
    reps = 8 if fast else 30
    burn, iters = (1500, 4000) if fast else (15000, 60000)
    B_list = [16, 50]                                 # 50 = floor(0.1 * 506)
    variants = []
    for preproc in ("icpt", "ceny", "stdy"):          # posterior-scale sweep (simulated)
        variants.append(run_variant(f"{preproc}-post", preproc, True, B_list, reps, burn, iters))
        print_variant(variants[-1])
    # unscaled-target regime, documented analytically (everything, incl. DQ+exact, is
    # unstable there -- it cannot be the paper's Table-3 regime)
    variants.append(run_variant("icpt-unscaled", "icpt", False, B_list, reps, burn, iters,
                                simulate_it=False))
    print_variant(variants[-1])

    # PRE-REGISTERED primary rule: both stated premises of the claim's regime must
    # reproduce -- DQ+exact accurate AND LR+WS divergent, at both batch sizes.
    def qualifies(v):
        return all(
            (not v["cells"][str(B)]["DQ+exact"]["diverged"])
            and v["cells"][str(B)]["DQ+exact"]["exact_cov_err"] < 1.0
            and v["cells"][str(B)]["LR+WS"]["diverged"]
            for B in B_list)

    def ct_gap(v):
        e = v["cells"]["16"]["CT"]["exact_cov_err"]
        return abs(e - PAPER_TABLE3_LOG["16"]["CT"]) if np.isfinite(e) else np.inf

    qual = [v for v in variants if v["nscale"] and qualifies(v)]
    primary = min(qual, key=ct_gap) if qual else None
    controls = {}
    if primary is not None:
        wsv = run_variant(primary["tag"] + "-wellspec", primary["preproc"], True, B_list,
                          reps, burn, iters, control="wellspec-gauss")
        print_variant(wsv)
        controls["wellspec_gauss"] = wsv
        wor = run_variant(primary["tag"] + "-worepl", primary["preproc"], True, B_list,
                          reps, burn, iters, replacement=False)
        print_variant(wor)
        controls["without_replacement"] = wor

    ct_all = {v["tag"]: {B: dict(rho=v["cells"][str(B)]["CT"]["rho"],
                                 exact=v["cells"][str(B)]["CT"]["exact_cov_err"],
                                 diverged=v["cells"][str(B)]["CT"]["diverged"])
                         for B in map(str, B_list)} for v in variants}

    out = dict(seed=SEED, fast=fast, reps=reps, burn=burn, iters=iters, B_list=B_list,
               paper_table3_log=PAPER_TABLE3_LOG, paper_ct_ci=PAPER_CT_CI,
               diverge_thresh=DIVERGE_THRESH, finite_thresh=FINITE_THRESH,
               variants=variants, controls=controls,
               primary_tag=(primary["tag"] if primary is not None else None),
               primary=primary, ct_across_variants=ct_all,
               elapsed_s=round(time.time() - t0, 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "c5_falsification.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote outputs/c5_falsification.json  (elapsed {out['elapsed_s']}s)", flush=True)
    # in local-mode orx the run log is the only evidence channel: emit the raw JSON so the
    # committed artifact can be recovered verbatim from the formal run's log
    print("\n===BEGIN c5_falsification.json===", flush=True)
    print(json.dumps(out, default=float), flush=True)
    print("===END c5_falsification.json===", flush=True)
    return out


if __name__ == "__main__":
    run_all(fast="--fast" in sys.argv)
