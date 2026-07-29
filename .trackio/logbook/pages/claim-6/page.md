# Claim 6 — Proposition B.1: SGLD-with-momentum; κ→0 recovers Eq 15

**Exact claim:** the momentum proxy's stationary covariance satisfies Eq B.5; it recovers the non-momentum result (Eq 15) as κ→0.

## Verdict: VERIFIED

### Evidence
**Symbolic certificates** (`symbolic_verify.py`):
- `verify_momentum_kappa0` ✅ — substituting κ=0 into Eq B.5 reduces it identically to Eq 15 (algebraic identity over symmetric matrices).
- `verify_momentum_augmented` ✅ — Eq B.5 independently reconstructed from the augmented-state (ψ,ν) discrete Lyapunov; its (0,0) block at κ=0 equals the non-momentum Lyapunov solution.

**Numerical κ→0 limit** (logistic D=4; momentum covariance via `sgduq_core.sigma_psi_momentum`):

| κ | rel err vs non-momentum |
|---|---|
| 0.50 | 0.993 |
| 0.20 | 0.248 |
| 0.10 | 0.110 |
| 0.05 | 0.052 |
| 0.02 | 0.020 |
| 0.01 | 0.0100 |
| 0.00 | **5.8×10⁻¹⁶** |

The momentum covariance converges **monotonically** to the non-momentum covariance as κ→0, reaching machine precision at κ=0 (5.8×10⁻¹⁶). Fitted κ-exponent ≈ 1.15 (linear in κ).

**Negative control:** at κ=0.5 the covariance differs by 0.99 (clearly distinct), confirming the extension is non-trivial and the κ→0 recovery is meaningful.

**Code:** `verify_sgduq.claim6`, `sgduq_core.sigma_psi_momentum`, `symbolic_verify.verify_momentum_*`. **Raw:** `outputs/c6_propB1_momentum.json`. Confidence: **HIGH**.
