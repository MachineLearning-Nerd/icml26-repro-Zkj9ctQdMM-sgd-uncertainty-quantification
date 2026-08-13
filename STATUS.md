# Reproduction status

## Paper

- **Title:** Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo
- **Authors:** Yu Wang, Jie Ding, and Jonathan H. Huggins
- **arXiv:** [2606.00293](https://arxiv.org/abs/2606.00293)
- **OpenReview:** [Zkj9ctQdMM](https://openreview.net/forum?id=Zkj9ctQdMM)

## Decision

- **Overall paper-level status:** `INCONCLUSIVE`
- **Scoped reproduction gate:** `PASS`
- **Claim 5:** `FALSIFIED_AS_REGISTERED`
- **Raw suite:** 5 `VERIFIED`, 1 `FALSIFIED`; see [`outputs/verdict.json`](outputs/verdict.json)
- **Conservative gate manifest:** [`publication_gate.json`](publication_gate.json)

The scoped evidence is substantial, but it does not independently establish every theorem for every assumption regime. In particular, Claims 1 and 3 rely on symbolic derivations plus finite numerical corroboration whose fitted rates are Monte Carlo-noise-limited. Claim 4 is a bounded algorithmic reproduction, not a universal benchmark claim.

## Evidence-bearing state

- Full-Boston Claim 5 evidence: [`outputs/c5_falsification.json`](outputs/c5_falsification.json)
- Deterministic Claim 5 gate: [`repro/src/check_claim5_falsification.py`](repro/src/check_claim5_falsification.py)
- Fixed suite command: `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py`
- Formal backend recorded in the evidence log: Hugging Face `cpu-upgrade`, no GPU
- Claim 5 run: seed `20260729`, 30 chains per cell, 15,000 burn-in and 60,000 measured iterations

## Repository policy

- **Maintainer identity:** MachineLearning-Nerd
- **Canonical branch:** `main`
- **Historical experiment branches:** documented in [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md), then removed from the remote after their roles were recorded
- **Official paper code:** [wangyu1369/large-sample-sgmcmc-uq](https://github.com/wangyu1369/large-sample-sgmcmc-uq)
- **This repository:** independent reproduction and audit, not an official author release

See [`README.md`](README.md) for the complete claim ledger, producer paths, citation, thank-you note, and reproduction instructions.
