# ICML 2026 — SGD uncertainty quantification

Independent reproduction and claim audit for [*Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo*](https://arxiv.org/abs/2606.00293) ([OpenReview](https://openreview.net/forum?id=Zkj9ctQdMM)).

## Status at a glance

**Paper-level assessment: INCONCLUSIVE.** The repository contains strong, scoped evidence for the six registered items, including a reproducible falsification of the broad Claim 5 wording. That is not the same as independently proving every theorem under all of its assumptions.

| Item | Registered statement | Evidence status | What produces the result |
|---|---|---|---|
| C1 — Theorem 4.1 | Relative covariance error is $O(\sqrt{\lambda})$ | **VERIFIED_CONDITIONAL** | Symbolic Eq. 15/rate certificate plus logistic-regression numerics and a mis-centering control |
| C2 — Theorem 4.3 | Exact proxy minibatch-noise covariance, Eq. 16 | **VERIFIED_CONDITIONAL** | Symbolic derivation plus Monte Carlo formula check and constant-noise negative control |
| C3 — Theorem 4.5 / Cor. 4.6 | $W_2(\pi_\theta,\pi_\psi) \le A\lambda/B$ | **VERIFIED_CONDITIONAL** | True-vs-proxy Wasserstein calculation, bound check, and mis-centered control |
| C4 — Algorithm 1 | Two-stage DQ+exact tuning improves covariance targeting | **REPRODUCED_SCOPED** | M=200 sandwich stage, exact-noise solve, preconditioned chain, and competitor comparison |
| C5 — Table 3 | DQ+exact is accurate while CT and constant-noise competitors diverge | **FALSIFIED_AS_REGISTERED** | Paper-table transcription plus independent 506-row Boston replay, exact stability operator, 30-chain simulation, and 8 checks |
| C6 — Proposition B.1 | Momentum covariance Eq. B.5 recovers Eq. 15 as \(\kappa\to0\) | **VERIFIED_CONDITIONAL** | Symbolic augmented-state Lyapunov derivation and numerical \(\kappa\)-sweep |

The raw evaluator records five `VERIFIED` entries and one `FALSIFIED` entry in [`outputs/verdict.json`](outputs/verdict.json). [`publication_gate.json`](publication_gate.json) is the conservative paper-level wrapper: the scoped evidence gate passes, while the overall assessment remains `INCONCLUSIVE`.

## What Claim 5 says—and what the evidence says

The registered wording treats the Table 3 comparison as if both continuous-time (CT) and constant-noise competitors diverge. The paper’s own log-loss Table 3 reports finite CT covariance errors of **0.247** at $B=16$ and **0.589** at $B=50=\lfloor0.1N\rfloor$. The independent replay uses all **506 Boston rows and 13 features plus an intercept**, with no stability clamps:

| Method | (B=16): spectral radius / exact / simulated median | (B=50): spectral radius / exact / simulated median |
|---|---:|---:|
| DQ+exact | 0.906 / 0.015 / 0.041 | 0.737 / 0.015 / 0.024 |
| CT | 0.896 / 0.073 / 0.077 | 0.703 / 0.155 / 0.155 |
| LR+WS | 1.564 / ∞ / 6.3×10⁴ | 1.707 / ∞ / 2.0×10⁴ |
| DQ+const | 3.156 / ∞ / ∞ | 2.918 / ∞ / ∞ |

The exact second-moment operator proves the LR+WS and DQ+const instability; 30/30 CT chains remain finite at both batch sizes. This falsifies the broad registered conjunction, while preserving the narrower paper observation that LR+WS becomes unstable. See the [full Claim 5 report](reports/claim5-falsification/report.md).

## How each claim is produced

The claim result is never inferred from a filename or a single plot. Each row below names the producer, the raw artifact, and the control or limitation that bounds the interpretation.

| Claim | Producer path | Raw artifact | Control / interpretation boundary |
|---|---|---|---|
| C1 | `repro/src/symbolic_verify.py` reconstructs Eq. 15 and the $\sqrt{\lambda}$ mechanism; `repro/src/verify_sgduq.py::claim1` runs logistic SGD across four step sizes | [`outputs/c1_thm4.1_sqrt_lambda.json`](outputs/c1_thm4.1_sqrt_lambda.json) | Mis-centered proxy has control exponent −0.113; numerical rate is MC-noise-limited, so the status is conditional on the symbolic assumptions |
| C2 | `symbolic_verify.verify_eq16`; `verify_sgduq.py::claim2`; `sgduq_core.noise_cov_eq16` and `mc_noise_cov` | [`outputs/c2_thm4.3_exact_noise.json`](outputs/c2_thm4.3_exact_noise.json) | Formula-vs-MC relative errors are 0.017–0.032; exact noise differs from constant-noise by 0.046–0.078 |
| C3 | `verify_sgduq.py::claim3`; `sgduq_core.w2_gaussian` compares the true-loss stationary covariance with the proxy covariance | [`outputs/c3_thm4.5_W2_bound.json`](outputs/c3_thm4.5_W2_bound.json) | Bound holds with fitted $A=1.56$; fitted rates are MC-noise-limited and the mis-centered proxy is the negative control |
| C4 | `verify_sgduq.py::claim4`; `sgduq_core.solve_lambda_eigenbasis` performs the offline sandwich solve and stage-2 chain comparison | [`outputs/c4_algorithm1_two_stage.json`](outputs/c4_algorithm1_two_stage.json) | Scoped to a small misspecified heteroscedastic linear model with $M=200,B=16,D=5$; it is not a universal performance claim |
| C5 | `repro/src/claim5_falsify.py` builds the full Boston regime; `check_claim5_falsification.py` evaluates conditions A–H; `verify_sgduq.py::claim5` records the verdict | [`outputs/c5_falsification.json`](outputs/c5_falsification.json), [`outputs/c5_boston_table3.json`](outputs/c5_boston_table3.json) | Paper-table evidence, exact operator, simulation, detector, well-specified, cross-check, and protocol controls all agree; the result targets the registered wording only |
| C6 | `symbolic_verify.verify_momentum_*`; `verify_sgduq.py::claim6`; `sgduq_core.sigma_psi_momentum` sweeps $\kappa$ to zero | [`outputs/c6_propB1_momentum.json`](outputs/c6_propB1_momentum.json) | At $\kappa=0$, relative error is 5.8×10⁻¹⁶; $\kappa=0.5$ is a non-trivial negative control |

The suite orchestrator is [`repro/src/verify_sgduq.py`](repro/src/verify_sgduq.py). It writes the six per-claim JSON files and exits nonzero if any claim is unresolved. The Claim 5 checker is intentionally stricter: it exits nonzero if the falsification evidence is absent.

## Historical branch roles

The historical experiment branches are documented in [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). They are deliberately not retained as live branches after cleanup; the final repository uses only `main` as its reader-facing publication surface.

| Historical branch | Role | Outcome |
|---|---|---|
| `orx/baseline` | Initial environment, Boston data, and early reproduction scaffold | Historical toy baseline; not the release evidence |
| `orx/faithful-symbolic-numeric-repro` | Clean-room symbolic identities, exact discrete-time numerics, initial six-claim suite | Established the mechanisms; Claim 5 was not yet full-Boston |
| `orx/faithful-pip-repro` | Pinned pip/UV reproduction of all six checks | Useful cumulative baseline; Claim 5 still used a six-feature toy and did not support the final falsification |
| `orx/claim5-falsification-full-boston` | Full 506×13 Boston replay, exact second-moment operator, controls, and raw evidence | Produced the Claim 5 falsification evidence |
| `orx/claim5-release-validation` | Immutable release rerun of the evidence-bearing state | Revalidated 8/8 Claim 5 checks and claims 1–4/6 |
| `release-claim5-falsification` | Clean-clone publication-gate handoff | Merged release handoff into the publication surface |
| `main` | Final reader-facing README, reports, notebook, figures, raw JSON, and gate metadata | Only live branch after cleanup |

`orx/*` names identify historical experiment stages, not supported user workflows. Their roles and source commit IDs are preserved in the audit file even though the remote branches are removed to prevent stale or ambiguous entry points.

## Reproduce or inspect

The evidence-bearing command is intentionally pinned:

```bash
pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0
python repro/src/verify_sgduq.py
```

The full suite takes about 20 minutes on the recorded `cpu-upgrade` worker because Claim 5 runs 30 chains per cell. The committed raw JSON is available for inspection without rerunning it. A bounded deterministic Claim 5 check is:

```bash
python repro/src/check_claim5_falsification.py
```

To view the notebook:

```bash
uvx marimo edit notebooks/claim5_falsification.py
uvx marimo run notebooks/claim5_falsification.py
```

The source-level evidence map is in [`docs/PRIMARY_SOURCE_MAP.md`](docs/PRIMARY_SOURCE_MAP.md), and the release decision is in [`STATUS.md`](STATUS.md) and [`GATE_READY.md`](GATE_READY.md).

## Citation

```bibtex
@misc{wang2026accuratelargesample,
  author       = {Yu Wang and Jie Ding and Jonathan H. Huggins},
  title        = {Accurate Large-sample Uncertainty Quantification using
                  Stochastic Gradient Markov Chain Monte Carlo},
  year         = {2026},
  eprint       = {2606.00293},
  archivePrefix = {arXiv},
  url          = {https://arxiv.org/abs/2606.00293}
}
```

If this audit or its figures are useful, please cite the paper above and the relevant evidence files in this repository. This repository is an independent reproduction/audit maintained by **MachineLearning-Nerd**; it is not the authors’ official implementation.

## Thank you

Thank you to **Yu Wang, Jie Ding, and Jonathan H. Huggins** for developing the discrete-time uncertainty-quantification framework and for presenting the equations, assumptions, and experimental comparisons clearly enough to audit. The reproduction is intended as a respectful, evidence-based companion to the paper, including the parts that remain conditional or inconclusive.

The authors’ official code repository is [wangyu1369/large-sample-sgmcmc-uq](https://github.com/wangyu1369/large-sample-sgmcmc-uq). This audit’s implementation was reconstructed independently and should not be treated as an official release.

## Maintainer

Repository and publication-surface commits are maintained under the **MachineLearning-Nerd** GitHub identity. See [`STATUS.md`](STATUS.md) for the current branch, commit, and gate state.
