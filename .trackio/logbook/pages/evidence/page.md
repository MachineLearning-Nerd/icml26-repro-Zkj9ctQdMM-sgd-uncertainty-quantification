# Evidence & method

## Fixed command & pinned environment (identical on every node)
```
pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py
```
- Python 3.12 (HF `python:3.12` image); numpy 2.5.1, scipy 1.18.0, sympy 1.14.0 (pinned; match `uv.lock`).
- Deterministic seed base `20260724` (every stochastic call derives a sub-seed).
- Git: winning branch `orx/faithful-pip-repro` @ `f9283d6`; baseline `orx/baseline` @ `923a9df` (toy, frozen).
- HF run `c626a479` on `cpu-upgrade` (1 worker), 33 s; local re-run 40 s (1 core).

## Source audit
- Paper HTML retrieved 2026-07-24 from `https://ar5iv.labs.arxiv.org/html/2606.00293` (browser User-Agent), **SHA-256 `56ad1873de185c393103c1ec294fabca67e8febd1bf942cae641a0c007f2ff5d`**.
- Theorem/section anchors and exact quantifiers: see `.openresearch/artifacts/source_audit.md` and [Claims](#/claims).

## Method (clean-room)
The reproduction reconstructs the paper's discrete-time proxy theory from first principles — **no paper code was used** (the paper's repo `wangyu1369/large-sample-sgmcmc-uq` exists but was not consulted for implementation):
- `repro/src/symbolic_verify.py` — 5 sympy identities (machine-checkable): Eq 15 from the linear-Gaussian Lyapunov map; Eq 16 noise covariance; Eq B.5 κ→0 reduction; augmented-(ψ,ν) momentum reconstruction; the √λ rate mechanism (W₂=O(λ), ‖Σ‖=O(λ) ⇒ rel err O(√λ)). All 5 PASS.
- `repro/src/sgduq_core.py` — exact discrete-time numerics: Eq 15+16 solved as one d²×d² linear system (no iteration); Eq B.5 via augmented Lyapunov; Algorithm 1 (offline sandwich → solve for Λ); 2-Wasserstein (Gaussian/Bures); fast minibatch SGD chains; real-Boston sandwich.
- `repro/src/verify_sgduq.py` — 6 claim checks, each with a symbolic certificate + numerical evidence + a negative control; writes `outputs/verdict.json` + per-claim JSON and **exits nonzero** if any VERIFIED-required check fails.

## Raw outputs (regenerable from the fixed command)
`outputs/verdict.json` (summary), `outputs/c{1..6}_*.json` (per-claim metrics), and `outputs/c5_falsification.json` (full falsification evidence: every variant, cell, control, and 30 per-chain errors — extracted verbatim from run `95bc366e`'s log, which prints it between `===BEGIN/END c5_falsification.json===` markers). Deterministic gates: `repro/src/verify_sgduq.py` (exit nonzero unless all six claims resolve) and `repro/src/check_claim5_falsification.py` (exit nonzero unless the claim-5 falsification is present with all 8 conditions).

## Claim-5 falsification method (added on `orx/claim5-falsification-full-boston`)
- `repro/src/claim5_falsify.py` — full 506×13 Boston; Algorithm-1 tunings with **no stability clamps** (DQ+exact / DQ+const / LR+WS / CT + three sensitivity forms); **exact second-moment operator** T = E[(I−ΛH_B)⊗(I−ΛH_B)] (196×196, closed form; ρ(T)≥1 ⇔ divergence — assumption-free for the quadratic log loss) **plus** 30 real-minibatch SGD chains per cell; protocol sweep with a pre-registered primary rule; positive/detector/assumption controls; machine-precision closed-form cross-check (3×10⁻¹⁵).

## Compute log
| Run | Branch@commit | Backend | Result | Time |
|---|---|---|---|---|
| **release validation** | **orx/claim5-release-validation@94def38** | **HF cpu-upgrade** | **5 VERIFIED + claim 5 FALSIFIED, 8/8 checks, exit 0** | **17m52s** |
| **95bc366e** | **orx/claim5-falsification-full-boston@8bd80fe** | **HF cpu-upgrade** | **5 VERIFIED + claim 5 FALSIFIED, 8/8 checks, exit 0** | **19 min** |
| c626a479 | orx/faithful-pip-repro@f9283d6 | HF cpu-upgrade | 6/6 VERIFIED (claim 5 later re-judged TOY) | 33 s |
| local | orx/faithful-pip-repro@f9283d6 | local CPU | 6/6 VERIFIED | 40 s |
| (historical) | orx/baseline@923a9df | — | toy 2/6 | — |

## Assumptions kept satisfied
Experiments use logistic regression (convex, smooth, μ-strongly-convex with L2 reg ⇒ Assumptions A–C) and linear regression on Boston housing (quadratic loss ⇒ proxy exact; misspecification is in the error structure, the paper's setting). Step sizes satisfy λ < 1/(2L).
