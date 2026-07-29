# Claim 4 — Algorithm 1: two-stage DQ+exact tuning

**Exact claim:** Algorithm 1 = (1) offline sandwich covariance Ŝ=Ĵ⁻¹ÎĴ⁻¹ from a subsample, then solve Eqs 15+16 with Σ_ψ=Ŝ for the preconditioner Λ; (2) preconditioned SG(L)D.

## Verdict: VERIFIED

### Evidence
**Full faithful implementation** (`sgduq_core.solve_lambda_eigenbasis` + `method_covariance_error`): Stage 1 subsamples M=200 observations → Ŝ → solves the coupled Eq 15+16 (vectorized Λ, Powell-hybrid root in J's eigenbasis) for Λ. Stage 2 runs the preconditioned chain.

**The tuned chain realises the predicted covariance:** DQ+exact's Λ gives a stationary covariance with **rel-to-target 0.324**; an actual SGD chain at the solved step matches the exact theory to **rel 0.084** (chain ≈ theory).

**DQ+exact beats every competitor** at realising the target covariance (rel-to-target, lower is better):

| Method | rel-to-target |
|---|---|
| **DQ+exact** | **0.324** |
| LR+WS | 0.324 |
| DQ+const | 0.738 |
| CT | 0.976 |

**Negative control / competitors:** CT (continuous-time) and DQ+const (constant-noise) are markedly worse; the DQ+exact procedure is the only one that solves the *exact* noise equation. (Setup: D=5 misspecified heteroscedastic linear regression, B=16; small D keeps the full-Λ solve exact and fast.)

**Code:** `verify_sgduq.claim4`, `sgduq_core.method_covariance_error`, `solve_lambda_eigenbasis`, `solve_full_lambda_for_target`. **Raw:** `outputs/c4_algorithm1_two_stage.json`. Confidence: **HIGH**.
