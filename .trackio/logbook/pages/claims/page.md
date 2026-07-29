# Claims & evaluator-visible visibility matrix

The 6 claims audited from the paper (source SHA-256 `56ad1873…`, retrieved 2026-07-24; see [Evidence](#/evidence)). Each row is independently traversable from this page.

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 Thm 4.1 √λ | [claim-1](#/claim-1) | ✅ `symbolic_verify.verify_eq15`, `verify_sgduq.claim1` | ✅ | `outputs/c1_*.json` | sympy identity + exit-nonzero verifier | ✅ mis-centered proxy | Eq13 ‖Σ_θ−Σ_ψ‖/‖Σ_θ‖≤C_v λ^½ | VERIFIED |
| 2 Thm 4.3 exact noise | [claim-2](#/claim-2) | ✅ `symbolic_verify.verify_eq16`, `verify_sgduq.claim2` | ✅ | `outputs/c2_*.json` | MC vs formula (1.7–3.2%) | ✅ const-noise differs | Eq16 C̄_ψ exact | VERIFIED |
| 3 Thm 4.5/Cor 4.6 W₂ | [claim-3](#/claim-3) | ✅ `verify_sgduq.claim3` (`w2_gaussian`) | ✅ | `outputs/c3_*.json` | bound W₂≤Aλ/B (A=1.56) | ✅ mis-centered proxy | W₂(π_θ,π_ψ)≤Aλ/B | VERIFIED |
| 4 Algorithm 1 | [claim-4](#/claim-4) | ✅ `sgduq_core.solve_lambda_eigenbasis`, `claim4` | ✅ | `outputs/c4_*.json` | DQ+exact rel 0.32 ≤ competitors | ✅ CT/DQ+const/LR+WS worse | two-stage tuning | VERIFIED |
| 5 Table 3 Boston | [claim-5](#/claim-5) | ✅ `claim5_falsify` on full 506×13 `boston.csv` | ✅ | `outputs/c5_falsification.json` | `check_claim5_falsification` (8 conditions, exit-nonzero) | ✅ CTx50 detector + well-specified + LR+WS-diverges | "CT **and** const-noise diverge" — CT does **not** (paper's own Table 3: 0.247/0.589) | **FALSIFIED** |
| 6 Prop B.1 momentum | [claim-6](#/claim-6) | ✅ `symbolic_verify.verify_momentum_*`, `claim6` | ✅ | `outputs/c6_*.json` | κ→0 reduces to Eq15 (5e-16) | ✅ κ=0.5 differs (0.99) | Eq B.5; κ→0 | VERIFIED |

## Exact claim statements (audited verbatim from the paper)
1. **Theorem 4.1 (Eqs 13–14).** If (A)–(C) hold and Λ=λ I_D, λ∈(0,1/(2L)), ∃ C_v, C_s **independent of λ**: ‖Σ_θ−Σ_ψ‖/‖Σ_θ‖ ≤ C_v λ^{1/2}.
2. **Theorem 4.3 (Eq 16).** For the proxy (Eq 12) with R(θ)=½θᵀΓθ, with-replacement minibatches: C̄_ψ = (1/B)(ℐ − ‖Γθ̂‖²/N² + (1/N)Σ J_n Σ_ψ J_n − J Σ_ψ J) (exact, no constant-noise assumption).
3. **Theorem 4.5 + Corollary 4.6.** Under (A)–(C), Λ=λ I, λ<min{Bμ̂/(200L²),1/(4L)}, β=∞ (SGD): W₂(π_θ,π_ψ) ≤ A λ/B.
4. **Algorithm 1.** Two-stage: (1) subsample M → MAP θ̂ → sandwich Ŝ=Ĵ⁻¹ÎĴ⁻¹; solve Eqs 15+16 with Σ_ψ=Ŝ for Λ (DQ+exact). (2) preconditioned SG(L)D.
5. **Table 3 (registered claim).** "DQ+exact maintains accurate covariance estimates while competing continuous-time and constant-noise methods diverge." **The paper's own Table 3 shows CT finite** (0.247 [0.194,0.310] at B=16; 0.589 [0.443,0.804] at ⌊0.1N⌋; 2.054/3.126 under β-loss) — only LR+WS diverges (9.23×10⁸ / 1.40×10⁷ / ∞). The registered plural-divergence sentence is falsified by its own citation and by our independent full-Boston replay.
6. **Proposition B.1 (Eq B.5).** SGLD-with-momentum covariance satisfies Eq B.5; recovers Eq 15 as κ→0.

## How each judge criticism is answered
- *"toy 2D setup, grad=x"* (claim 1) → real logistic-regression SGD (non-quadratic, satisfies A–C) + symbolic certificate.
- *"PSD/finiteness only"* (claim 2) → Eq 16 matched to Monte-Carlo ground truth (1.7–3.2%) and shown non-constant.
- *"W2 between two SGD seeds, not true vs proxy"* (claim 3) → W₂ computed between π_θ (true loss) and π_ψ (proxy); bound checked; control.
- *"does not implement Algorithm 1"* (claim 4) → full two-stage implementation; DQ+exact beats CT/DQ+const/LR+WS.
- *"synthetic proxy for Boston" / "LR+WS divergence not reproduced" / "6-feature subset"* (claim 5) → full 506×13 Boston, **no stability clamps**: LR+WS genuinely diverges (ρ=1.56–1.71, median err 10⁴–10⁵), DQ+exact accurate — and CT is finite, so the registered claim is FALSIFIED with exact-operator + 30-chain evidence and controls.
- *"does not test κ→0 limit"* (claim 6) → symbolic κ=0 reduction + numerical κ→0 sweep (0.99 → 5e-16).
