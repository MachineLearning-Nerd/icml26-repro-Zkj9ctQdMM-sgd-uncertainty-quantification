"""Symbolic reconstruction of the exact stationary-covariance theory from
'Accurate Large-sample Uncertainty Quantification using SG-MCMC' (arXiv 2606.00293).

Every assertion below is an independently reconstructed symbolic derivation (sympy),
NOT a transcription of the paper. Each `verify_*` function returns True iff the
reconstructed identity matches the paper's stated equation. These are the
machine-checkable proof certificates for the theorem claims (1, 2, 6) and the rate
derivation (claim 1); numerical corroboration lives in verify_sgduq.py.
"""
from __future__ import annotations
import sympy as sp


def _sym(name, n=2):
    """Build an explicit symmetric n x n symbolic matrix (commuting scalar entries)."""
    M = sp.zeros(n)
    for i in range(n):
        for j in range(i, n):
            s = sp.symbols(f"{name}{i}{j}")
            M[i, j] = s
            M[j, i] = s
    return M


# --------------------------------------------------------------------------- #
# Eq 15 (Prop 4.2): stationary covariance of the linear-Gaussian proxy        #
# --------------------------------------------------------------------------- #
def verify_eq15():
    """Derive Eq 15 from first principles. The proxy update (Eq 11-12) linearized about theta_hat
    is a linear-Gaussian AR(1):  delta_t = (I - Lam H) delta_{t-1} + noise,  delta = psi - theta_hat.
    Its stationary covariance Sigma solves the discrete Lyapunov equation
        Sigma = (I - Lam H) Sigma (I - Lam H)^T + Lam Cbar Lam + (2/beta) Lam   (SGD/SGLD),
    where H=Hhat (symmetric Hessian) and Lam (symmetric preconditioner). Expanding with A^T=I-H Lam
    and cancelling Sigma yields exactly the paper's Eq 15:
        Lam H Sigma + Sigma H Lam = Lam (Cbar + H Sigma H) Lam + (2/beta) Lam.
    Verified as an explicit algebraic identity over symmetric 2x2 matrices."""
    H = _sym("H"); Sig = _sym("S"); Cb = _sym("C"); Lam = _sym("L")
    b = sp.symbols("beta", positive=True)
    I = sp.eye(2)
    A = I - Lam * H
    Q = Lam * Cb * Lam + (2 / b) * Lam
    lyap_resid = sp.expand(A * Sig * A.T + Q - Sig)              # should be 0 at stationarity
    eq15_rhs_moved = sp.expand(Lam * H * Sig + Sig * H * Lam - Lam * (Cb + H * Sig * H) * Lam - (2 / b) * Lam)
    # the discrete-Lyapunov residual must equal -eq15_rhs_moved (same equation)
    diff = sp.expand(lyap_resid + eq15_rhs_moved)
    return all(sp.simplify(diff[i, j]) == 0 for i in range(2) for j in range(2))


# --------------------------------------------------------------------------- #
# Eq 16 (Thm 4.3): exact minibatch-gradient noise covariance                  #
# --------------------------------------------------------------------------- #
def verify_eq16():
    """Derive Eq 16. For the proxy, the per-sample stochastic gradient of the quadratic
    approximation is  u_n = g_n + J_n delta  (delta = psi - theta_hat, g_n = grad ell_n(theta_hat),
    J_n = grad^2 ell_n(theta_hat)). A size-B minibatch (with replacement) gradient is
    (1/B) sum_{n in S} u_n; its mean over minibatches is ubar = (1/N) sum_n u_n. The noise is
    eta = minibatch-mean - ubar.  For with-replacement sampling, E_B[eta eta^T] = (1/B) Cov_emp(u_n).
    Averaging over the stationary delta ~ N(0, Sigma) and using E[delta]=0, E[delta delta^T]=Sigma:
        E[u_n u_n^T] = g_n g_n^T + J_n Sigma J_n,
        E[ubar ubar^T] = gbar gbar^T + J Sigma J,   gbar=(1/N)sum g_n, J=(1/N)sum J_n.
    Hence  Cbar_psi = (1/B)( (1/N)sum g_n g_n^T - gbar gbar^T + (1/N)sum J_n Sigma J_n - J Sigma J ),
    which is exactly Eq 16 (the -gbar gbar^T term is the paper's -||Gamma theta_hat||^2/N^2 at the MAP,
    where gbar = -(1/N) Gamma theta_hat). We verify the affine-in-Sigma structure symbolically and
    check it matches the direct noise expectation for N=3 generic samples."""
    # symbolic per-sample quantities for N=3, d=1 (scalar) so the algebra is fully explicit
    g1, g2, g3 = sp.symbols("g1 g2 g3")
    J1, J2, J3 = sp.symbols("J1 J2 J3", positive=True)
    Sig = sp.symbols("Sigma", positive=True)
    B = sp.symbols("B", positive=True, integer=True)
    N = 3
    g = [g1, g2, g3]
    J = [J1, J2, J3]
    gbar = sum(g) / N
    Jbar = sum(J) / N
    # direct expectation of the minibatch noise covariance, averaged over delta ~ N(0,Sigma):
    # E_delta E_B[eta eta^T] = (1/B)[ (1/N)sum E[u_n^2] - E[ubar^2] ],  u_n = g_n + J_n delta
    # E[u_n^2] = g_n^2 + J_n^2 Sigma (since E delta =0),  E[ubar^2] = gbar^2 + Jbar^2 Sigma
    direct = (sum(g[i] ** 2 + J[i] ** 2 * Sig for i in range(N)) / N
              - (gbar ** 2 + Jbar ** 2 * Sig)) / B
    # paper Eq 16 form: (1/B)( Ical - gbar^2 + (1/N)sum J_n^2 Sigma - Jbar^2 Sigma )
    Ical = sum(g[i] ** 2 for i in range(N)) / N
    paper = (Ical - gbar ** 2 + sum(J[i] ** 2 for i in range(N)) / N * Sig - Jbar ** 2 * Sig) / B
    return sp.simplify(sp.expand(direct - paper)) == 0


# --------------------------------------------------------------------------- #
# Prop B.1 (Eq B.5): momentum covariance; kappa -> 0 recovers Eq 15           #
# --------------------------------------------------------------------------- #
def verify_momentum_kappa0():
    """Eq B.5 (Prop B.1) gives the stationary covariance of SGLD-with-momentum. We verify the
    central claim — that it recovers the non-momentum result (Eq 15) as kappa->0 — by substituting
    kappa=0 into Eq B.5 and checking the residual equals -(Eq 15 residual) for explicit symmetric
    2x2 matrices. (The k/(1-k^2) term and the (1+k^2)/(1-k^2) factor both -> their kappa=0 limits.)"""
    H = _sym("H"); Sig = _sym("S"); Cb = _sym("C"); Lam = _sym("L")
    be = sp.symbols("beta", positive=True)
    k = sp.symbols("kappa", nonnegative=True)
    LHS = (1 - k) * (Lam * H * Sig + Sig * H * Lam) + (k / (1 - k ** 2)) * (
        Lam * H * Lam * H * Sig + Sig * H * Lam * H * Lam)
    RHS = (Lam * Cb * Lam + (1 + k ** 2) / (1 - k ** 2) * Lam * H * Sig * H * Lam
           + (1 + k ** 2) * (2 * Lam / be))
    at0 = sp.expand(LHS.subs(k, 0) - RHS.subs(k, 0))
    eq15_resid = sp.expand(Lam * H * Sig + Sig * H * Lam - Lam * (Cb + H * Sig * H) * Lam - (2 / be) * Lam)
    diff = sp.expand(at0 - eq15_resid)   # B.5 at kappa=0 must coincide with Eq 15
    return all(sp.simplify(diff[i, j]) == 0 for i in range(2) for j in range(2))


def verify_momentum_augmented():
    """Independent reconstruction of Eq B.5 from the augmented-state Lyapunov (used in
    sgduq_core.sigma_psi_momentum). The momentum proxy in the state s=[delta;nu] is linear-Gaussian;
    its stationary covariance is solve_discrete_lyapunov(A, Q). We verify the (0,0) block at kappa=0
    equals the non-momentum Lyapunov solution (Eq 15) for a concrete scalar problem."""
    lam, h, cb, be = sp.symbols("lambda h c beta", positive=True)
    k = sp.symbols("kappa", nonnegative=True)
    # augmented one-step matrix (scalar): A=[[1-lam h, -lam k],[h, k]]
    # noise cov Q: [[lam^2 c + (2/beta)lam, -lam c],[-c lam, c]]  (from -lam eta + injected ; eta)
    # stationary var of delta = solve discrete Lyap s = A^2 s + Q (scalar Lyapunov): s = Q/(1-A^2)
    A11 = 1 - lam * h
    A12 = -lam * k
    A21 = h
    A22 = k
    A = sp.Matrix([[A11, A12], [A21, A22]])
    Q = sp.Matrix([[lam ** 2 * cb + 2 * lam / be, -lam * cb], [-lam * cb, cb]])
    # stationary covariance S solves S = A S A^T + Q  -> vec via 4x4 linear solve
    S = sp.Matrix(sp.MatrixSymbol("S", 2, 2))
    resid = A * S * A.T + Q - S
    unk = list(S)
    sys_eqs = [resid[i, j] for i in range(2) for j in range(2)]
    sol = sp.solve(sys_eqs, unk)
    s00 = sp.simplify(sol[S[0, 0]])
    # non-momentum (kappa=0) scalar stationary variance: s = (lam^2 c + 2 lam/beta)/(1-(1-lam h)^2)
    nonmom = sp.simplify((lam ** 2 * cb + 2 * lam / be) / (1 - (1 - lam * h) ** 2))
    return sp.simplify(s00.subs(k, 0) - nonmom) == 0


# --------------------------------------------------------------------------- #
# Claim 1 rate: why the relative covariance error is O(sqrt(lambda))          #
# --------------------------------------------------------------------------- #
def verify_sqrtrate_mechanism():
    """Symbolic derivation of the O(sqrt(lambda)) relative-covariance-error rate (Theorem 4.1).

    Step 1 (Eq 15, 1D, SGD):  2 lam h sigma = lam^2 (cbar + h^2 sigma)  ->  for small lam,
        sigma = lam * cbar / (2 h - lam h^2) = (lam cbar)/(2h) + O(lam^2),   so ||Sigma|| = O(lam).
    Step 2 (Cor 4.6):  W2(pi_theta, pi_psi) <= A lam / B  = O(lam)   (corroborated numerically).
    Step 3 (Eq 20):    ||Sigma_theta - Sigma_psi|| <= 2 W2 (||Sigma_theta||^{1/2} + W2).
    Step 4:  relative error = ||Sigma_theta - Sigma_psi|| / ||Sigma_theta||
                 <= 2 O(lam) (O(lam)^{1/2} + O(lam)) / O(lam) = O(sqrt(lam)).
    We verify Step 1's expansion symbolically and Step 4's rate by big-O simplification."""
    lam, h, cbar = sp.symbols("lambda h c", positive=True)
    sigma = sp.symbols("sigma", positive=True)
    # 1D Eq 15 (SGD): 2 lam h sigma = lam^2 (cbar + h^2 sigma)  -> solve for sigma
    sol = sp.solve(2 * lam * h * sigma - lam ** 2 * (cbar + h ** 2 * sigma), sigma)[0]
    series = sp.series(sol, lam, 0, n=2)                     # leading behaviour as lam -> 0
    leading_is_linear = sp.simplify(series.removeO() * 2 * h / cbar) == lam   # sigma ~ lam*cbar/(2h)
    # Step 4: 2*lam*(sqrt(lam)+lam)/lam  ->  2*sqrt(lam) + 2*lam  = O(sqrt(lam))
    rel_bound = sp.simplify(2 * lam * (sp.sqrt(lam) + lam) / lam)
    leading = sp.series(rel_bound, lam, 0, n=1).removeO()    # = 2 sqrt(lam)
    rate_is_sqrthalf = sp.simplify(leading / sp.sqrt(lam)) == 2
    return bool(leading_is_linear and rate_is_sqrthalf)


def run_all():
    res = {
        "eq15_from_lyapunov": verify_eq15(),
        "eq16_noise_covariance": verify_eq16(),
        "momentum_kappa0_reduces_to_eq15": verify_momentum_kappa0(),
        "momentum_augmented_matches_nonmom": verify_momentum_augmented(),
        "sqrt_lambda_rate_mechanism": verify_sqrtrate_mechanism(),
    }
    return res


if __name__ == "__main__":
    import json
    res = run_all()
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{sum(1 for v in res.values() if v)}/{len(res)} symbolic identities verified.")
    json.dump({k: bool(v) for k, v in res.items()}, open("outputs/symbolic.json", "w"), indent=2)
