# Claim 5 falsification — full Boston housing reproduction

This repository reproduces Claim 5 from [*Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo*](https://arxiv.org/abs/2606.00293) ([OpenReview](https://openreview.net/forum?id=Zkj9ctQdMM)).

**Assessment: FALSIFIED as registered.** The registered claim says DQ+exact stays accurate while both continuous-time (CT) and constant-noise competitors diverge. The paper’s own Table 3 reports finite CT covariance errors, **0.247** at \(B=16\) and **0.589** at \(B=\lfloor0.1N\rfloor\). Our independent full-data replay agrees: CT remains stable with exact/simulated errors **0.073/0.077** and **0.155/0.155**, while LR+WS and DQ+const cross the exact second-moment stability boundary.

The replay uses all **506 rows and 13 Boston features** (plus an intercept), not the earlier six-feature subset. It fixes NumPy/SciPy/SymPy versions, seeds 30 chains per cell, runs 75,000 iterations per chain, and uses Hugging Face `cpu-upgrade` only. No GPU was used. The result does not say the paper’s narrower prose is wrong: Section 6.1 itself says only LR+WS becomes unstable. It falsifies the broader registered claim exactly as written.

[Read the illustrated report](reports/claim5-falsification/report.md) · [Open the tutorial notebook](notebooks/claim5_falsification.py) · [Inspect the canonical claim page](.trackio/logbook/pages/claim-5/page.md) · [Raw JSON](outputs/c5_falsification.json) · [Deterministic checker](repro/src/check_claim5_falsification.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-Zkj9ctQdMM-sgd-uncertainty-quantification/blob/main/notebooks/claim5_falsification.py)

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/faithful-pip-repro`](https://github.com/MachineLearning-Nerd/icml26-repro-Zkj9ctQdMM-sgd-uncertainty-quantification/tree/orx/faithful-pip-repro) | Cumulative six-claim symbolic/numeric baseline; Claim 5 used a six-feature subset | `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py` | Claim 5 remained toy-scale; LR+WS divergence absent | HF `cpu-upgrade`, 1m03s |
| [`orx/claim5-falsification-full-boston`](https://github.com/MachineLearning-Nerd/icml26-repro-Zkj9ctQdMM-sgd-uncertainty-quantification/tree/orx/claim5-falsification-full-boston) | Full 506×13 replay, exact stability operator, 30-chain simulation, controls, raw evidence, checker | `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py` | Claim 5 FALSIFIED; all eight falsification gates passed; cumulative claims 1–4 and 6 unchanged | HF `cpu-upgrade`, 19m09s |
| [`orx/claim5-release-validation`](https://github.com/MachineLearning-Nerd/icml26-repro-Zkj9ctQdMM-sgd-uncertainty-quantification/tree/orx/claim5-release-validation) | Freeze the evidence-bearing source state and rerun the cumulative release gate | `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py` | PASS: all eight Claim 5 gates; claims 1–4 and 6 preserved; exit 0 | HF `cpu-upgrade`, 17m52s |
| `main` (mirrors validated `master`) | Reader-facing README, report, notebook, figures, raw evidence, and Space source | Not run as an experiment (publication surface) | Fresh-clone checker, figure determinism, artifact integrity, compilation, and strict marimo validation passed | Local single-core validation only |

## Reproduce or inspect

The full formal command above takes about 20 minutes on `cpu-upgrade`. The raw result is committed so readers do not need to rerun it. A quick deterministic check is:

```bash
python repro/src/check_claim5_falsification.py
```

The checker exits nonzero unless the full-Boston regime is present, CT is stable, DQ+exact is accurate, LR+WS diverges, detector and well-specified controls behave correctly, the analytic/simulation routes agree, and the finding survives protocol sensitivities.

To read the notebook locally:

```bash
uvx marimo edit notebooks/claim5_falsification.py
uvx marimo run notebooks/claim5_falsification.py
```

## Original project note

Repro — Accurate Large-sample Uncertainty Quantification using SGD. OpenReview `Zkj9ctQdMM`; arXiv `2606.00293`. Owner: loop12pt.
