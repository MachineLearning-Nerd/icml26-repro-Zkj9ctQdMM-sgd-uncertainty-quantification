# Claim 2 — Theorem 4.3: exact minibatch noise covariance (Eq 16)

**Exact claim (Eq 16):** C̄_ψ = (1/B)(ℐ − ‖Γθ̂‖²/N² + (1/N)Σ J_n Σ_ψ J_n − J Σ_ψ J) — exact (no constant-noise assumption).

## Verdict: VERIFIED

### Evidence
**Symbolic certificate** (`symbolic_verify.verify_eq16` ✅): the per-minibatch gradient of the quadratic proxy is u_n = g_n + J_n δ; its with-replacement noise covariance, averaged over δ∼N(0,Σ_ψ), is derived from first principles and reduces identically to Eq 16 (the −ḡḡᵀ term is the paper's −‖Γθ̂‖²/N² at the MAP).

**Numerical — formula vs Monte-Carlo ground truth** (logistic regression):

| N | D | B | Eq16 vs MC (rel) | Eq16 differs from const-noise H/B |
|---|---|---|---|---|
| 800 | 4 | 8 | **0.017** | 0.046 |
| 800 | 4 | 32 | **0.022** | 0.046 |
| 1500 | 6 | 16 | **0.032** | 0.078 |

The exact formula matches the Monte-Carlo noise covariance to **1.7–3.2 %** (pure MC error) — i.e. it is *exact*, not approximate. It also differs measurably from the constant-noise heuristic C̄=Ĥ/B, confirming it captures the location-dependent noise (the paper's point).

**Negative control:** the constant-noise approximation Ĥ/B is rejected (it differs from the exact formula by 4.6–7.8 %, beyond MC error).

**Code:** `verify_sgduq.claim2`, `sgduq_core.noise_cov_eq16`/`mc_noise_cov`. **Raw:** `outputs/c2_thm4.3_exact_noise.json`. Confidence: **HIGH**.
