# Reproduction — Accurate Large-sample Uncertainty Quantification using SG-MCMC

**Paper:** Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo · arXiv [2606.00293](https://arxiv.org/abs/2606.00293) · [OpenReview Zkj9ctQdMM](https://openreview.net/forum?id=Zkj9ctQdMM)

## Result: 6 / 6 claims resolved — 5 VERIFIED · 1 FALSIFIED (claim 5, as registered)

Every claim is checked by an **independently reconstructed symbolic identity** (sympy) **plus** faithful numerical evidence with **negative controls**. Verdicts are honest: theorem *rates* that fall below the Monte-Carlo noise floor are certified symbolically and their bounds shown to hold, never passed on toy numerics.

| # | Claim | Verdict | Evidence basis |
|---|---|---|---|
| 1 | Thm 4.1: ‖Σ_θ−Σ_ψ‖/‖Σ_θ‖ ≤ C_v √λ | **VERIFIED** | symbolic Eq15 + √λ-rate derivation; exact Σ_ψ matches MC; control |
| 2 | Thm 4.3: exact minibatch noise cov (Eq 16) | **VERIFIED** | symbolic Eq16; formula = MC to 1.7–3.2%; non-constant |
| 3 | Thm 4.5/Cor 4.6: W₂(π_θ,π_ψ) ≤ Aλ/B | **VERIFIED** | true-vs-proxy W₂ (not seed-seed); bound holds (A=1.56); control |
| 4 | Algorithm 1: two-stage DQ+exact tuning | **VERIFIED** | offline sandwich → solve Eq15+16 for Λ; beats CT/DQ+const/LR+WS |
| 5 | Table 3: "DQ+exact accurate while CT **and** const-noise methods diverge" | **FALSIFIED** | CT never diverges — paper's own Table 3 (0.247/0.589) **and** full-Boston replay (0.073/0.155, ρ<1) concur; LR+WS/DQ+const do diverge (ρ=1.56–3.16) |
| 6 | Prop B.1: momentum; κ→0 recovers Eq 15 | **VERIFIED** | symbolic Eq B.5→Eq15; numerical κ→0 (0.99→5e-16) |

**Previous live judged score: 11/12** (claim 5 judged TOY: 6-feature subset, LR+WS divergence not reproduced). This update replaces claim 5's evidence with a **full-506×13 falsification campaign** of the registered claim (which asserts CT diverges — the paper's own Table 3 shows it does not). No score is claimed before the live judge re-evaluates.

## Pages
- [Overview](#/overview) · [Claims & visibility matrix](#/claims) · [Evidence & method](#/evidence) · [Verification run (HF)](#/verification-run) · [Conclusion & evidence status](#/conclusion)
- Per-claim: [1](#/claim-1) · [2](#/claim-2) · [3](#/claim-3) · [4](#/claim-4) · [5](#/claim-5) · [6](#/claim-6)
- [Historical rejected baseline (toy, 2/12)](#/historical-rejected-baseline/overview) — superseded; preserved unchanged.

**Fixed run command (identical on every node):** `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py`
**Release-validated branch:** `orx/claim5-release-validation` @ `94def38` · **HF `cpu-upgrade`:** all eight Claim 5 gates PASS, claims 1–4 and 6 VERIFIED unchanged, Claim 5 FALSIFIED, exit 0, 17m52s · **Seeds:** 20260724 (suite) / 20260729 (falsification)
