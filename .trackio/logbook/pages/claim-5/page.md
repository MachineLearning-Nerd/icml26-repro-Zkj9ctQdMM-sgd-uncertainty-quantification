# Claim 5 — Table 3: Boston housing — FALSIFIED (as registered)

**Registered claim:** "On the Boston housing dataset under strong model misspecification, the proposed DQ+exact method maintains accurate covariance estimates while competing **continuous-time and constant-noise methods diverge** (Table 3)."

## Verdict: FALSIFIED

The claim quantifies divergence over **both** the continuous-time (CT) and constant-noise method families. The CT conjunct is false:

1. **The paper's own Table 3 (log loss, covariance error) reports CT finite in every cell**: 0.247 (B=16, 95% CI [0.194, 0.310]) and 0.589 (B=⌊0.1N⌋, CI [0.443, 0.804]); β-loss cells 2.054 / 3.126 — all finite. Only **LR+WS** diverges (9.23×10⁸ / 1.40×10⁷; ∞ under β-loss). The claim contradicts its cited table.
2. **Independent full-Boston replay** (506×13, real `repro/data/boston.csv`, SHA-256 `ab16ba38…`, **no stability clamps** — the flaw of the earlier toy attempt): in the reproduced Table-3 regime CT does **not** diverge while LR+WS **does**.

### Replay table (primary variant `icpt-post`; 30 chains/cell; HF cpu-upgrade, exit 0)

| Method | ρ(T) B=16 | cov err B=16 (exact / sim median) | ρ(T) B=50 | cov err B=50 | paper Table 3 (log loss) |
|---|---|---|---|---|---|
| **DQ+exact** | 0.906 | **0.015 / 0.041** | 0.737 | **0.015 / 0.024** | 0.337 / 0.352 (finite) |
| **CT** | 0.896 | **0.073 / 0.077** | 0.703 | **0.155 / 0.155** | **0.247 / 0.589 (finite)** |
| LR+WS | **1.564** | **∞ / 6.3×10⁴** | **1.707** | **∞ / 2.0×10⁴** | 9.23×10⁸ / 1.40×10⁷ (diverges) |
| DQ+const | **3.156** | ∞ / ∞ | **2.918** | ∞ / ∞ | (Table 4: 0.98–0.997, worst) |

ρ(T) is the spectral radius of the **exact second-moment operator** T = E[(I−ΛH_B)⊗(I−ΛH_B)] of the real minibatch chain (the log loss is quadratic, so this is closed-form, assumption-free): ρ≥1 ⇔ divergence. Simulation = 30 independent real-minibatch SGD chains (75k iters, warm start θ̂), the paper's 30-repetition protocol. The two routes agree everywhere.

### Mechanism (why LR+WS diverges and CT does not)
The log loss ℓ = ½(y−θᵀz)² fixes the model noise at σ²=1; Boston's actual residual variance is **21.9** — the strong misspecification the claim conditions on. LR+WS trusts the model's σ²=1 (Eq 6), underestimates the gradient noise ≈22×, and solves for a Λ that crosses the real chain's stability boundary (ρ=1.56–1.71). CT uses the *empirical* noise Î/B (Eq 3) — misspecification-robust — and stays stable. DQ+exact additionally carries the exact Eq-12 noise: most accurate.

### Controls & cross-checks (all PASS; 8/8 in the deterministic verifier)
- **Positive control:** LR+WS diverges at both batch sizes — the claim's own regime is reproduced, so CT's finiteness is measured *inside* that regime.
- **Detector control:** CT's Λ×50 flagged divergent by both routes (ρ=52/226; 0/8 chains finite) — finiteness is not a detection failure.
- **Assumption control:** Gaussian covariates + well-specified unit-noise responses ⇒ LR+WS becomes stable *and accurate* (0.081): the Boston divergence is caused by misspecification, not by our LR+WS implementation.
- **Closed-form cross-check:** the exact solution Λ = 2SĤ(C+ĤSĤ)⁻¹ of Eq 11 reproduces the target to 3×10⁻¹⁵.
- **Robustness:** CT finite (ρ<1) at both batch sizes in every posterior-scale protocol variant (`icpt`/`ceny`/`stdy` preprocessing) and under without-replacement sampling.

### Honest sensitivities (documented, not hidden)
- Applying Eq 3 as a **literal asymmetric matrix** (`CT-asym`) does destabilize the discrete chain (ρ=1.39–2.46) — but that reading is inconsistent with the paper's own finite CT column, so it cannot be the paper's implementation; the eigenbasis application of the same equation matches the paper (0.073–0.155 vs 0.247–0.589).
- LR+WS with the *estimated* residual variance instead of the model σ²=1 stays finite (err≈1.37) — the paper-magnitude divergence requires the model-noise reading, which is also the reading under which the claim's own LR+WS catastrophe reproduces.
- The unscaled-target regime is rejected: there even DQ+exact diverges (ρ=1.63), contradicting the paper's finite DQ+exact column — it cannot be Table 3's regime.

**Code:** `repro/src/claim5_falsify.py` (release-validated branch `orx/claim5-release-validation` @ `94def38`). **Raw:** `outputs/c5_falsification.json` (extracted verbatim from the evidence-generating run log and revalidated in the immutable child). **Checker:** `repro/src/check_claim5_falsification.py` — exits **nonzero** if the falsification is absent (conditions A–H). **Command (fixed contract):** `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py`. Confidence: **HIGH** (paper's own table + two independent evidence routes + controls).
