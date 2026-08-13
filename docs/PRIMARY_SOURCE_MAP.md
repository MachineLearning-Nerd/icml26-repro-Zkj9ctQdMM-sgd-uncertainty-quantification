# Primary source map

## Paper sources

| Source | Use in this audit |
|---|---|
| [arXiv abstract and record](https://arxiv.org/abs/2606.00293) | Canonical title, authors, identifier, and paper landing page |
| [OpenReview record](https://openreview.net/forum?id=Zkj9ctQdMM) | ICML submission record and discussion context |
| [Official author code](https://github.com/wangyu1369/large-sample-sgmcmc-uq) | Attribution and comparison boundary; the implementation in this repository was reconstructed independently |

The six claim statements and equation references were transcribed into the committed Trackio claim pages under `.trackio/logbook/pages/claim-{1..6}/`. Those pages are an audit record; the paper links above remain the primary external sources.

## Local producer map

| Evidence layer | Producer | Consumed by |
|---|---|---|
| Symbolic identities | `repro/src/symbolic_verify.py` | Claims 1, 2, and 6; the Claim 1 rate mechanism |
| Exact numerical primitives | `repro/src/sgduq_core.py` | Claims 1–4 and 6; discrete Lyapunov, exact noise, Wasserstein, and tuning calculations |
| Six-claim orchestrator | `repro/src/verify_sgduq.py` | Writes `outputs/verdict.json` and the six per-claim JSON artifacts |
| Full-Boston campaign | `repro/src/claim5_falsify.py` | Writes `outputs/c5_falsification.json` for Claim 5 |
| Claim 5 deterministic gate | `repro/src/check_claim5_falsification.py` | Checks conditions A–H and exits nonzero if the falsification is absent |
| Report figures | `repro/src/make_claim5_report_figures.py` | Regenerates the SVGs under `reports/claim5-falsification/images/` from raw JSON |
| Reader notebook | `notebooks/claim5_falsification.py` | Presents the Claim 5 result interactively |

## Data and environment

- Boston input: `repro/data/boston.csv`; 506 observations and 13 features, with an intercept added by the producer code.
- Dependency lock: `uv.lock`.
- Reproduction environment: Python 3.11–3.12 contract in `pyproject.toml`; the recorded formal run used pinned NumPy 2.5.1, SciPy 1.18.0, and SymPy 1.14.0.
- Evidence manifest: [`publication_gate.json`](../publication_gate.json).

## Provenance boundary

`outputs/verdict.json` is the raw output of the cumulative evaluator and retains its evaluator labels. `publication_gate.json`, `STATUS.md`, and the README apply the conservative interpretation: symbolic and scoped numerical evidence is documented as conditional or scoped, Claim 5 is falsified only as registered, and the paper-level result is `INCONCLUSIVE`.
