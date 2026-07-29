# Conclusion & evidence status

## Target decision

**Claim 5: FALSIFIED as registered.** The exact target says DQ+exact stays accurate while continuous-time and constant-noise methods diverge. Under the stated Boston housing, log-loss, batch-size, and covariance-error regime:

- DQ+exact is accurate.
- LR+WS and DQ+const are unstable in the independent replay.
- CT is finite in the paper’s own Table 3 and in both independent analytic and simulation routes.

The contradiction is the CT conjunct. We do not alter the claim to a nearby statement and do not infer beyond this experiment.

## Evidence quality

- Full 506-row, 13-feature Boston data plus intercept.
- Exact second-moment operator for the real quadratic minibatch chain.
- Thirty seeded SGD chains per primary cell.
- Detector and well-specified controls.
- Closed-form tuning cross-check.
- Preprocessing and sampling sensitivities.
- Eight-condition deterministic checker that exits nonzero if the falsification is absent.
- Immutable HF `cpu-upgrade` release validation at commit `94def38`: all checks passed, claims 1–4 and 6 remained VERIFIED, exit 0.
- Fresh-clone checker, deterministic figure regeneration, artifact integrity, Python compilation, and strict marimo validation all passed.

## Limits

The replay’s LR+WS finite-horizon simulation errors are \(10^4\)–\(10^5\), below the paper’s \(10^7\)–\(10^9\), while the exact spectral radius still proves second-moment divergence. Chain length and overflow guards affect magnitude, not the stability verdict.

A literal asymmetric reading of Equation 3 destabilizes CT, but it is inconsistent with the paper’s finite CT column. The primary eigenbasis implementation reproduces the cited regime. The conclusion does not dispute that CT can be inaccurate elsewhere.

Claims 1 and 3 retain their previously documented Monte Carlo noise-floor limitations. They were regression-tested here, not newly adjudicated.

## Compute

Formal runs used one Hugging Face `cpu-upgrade` worker. The release validation took 17m52s. Local CPU was used only for bounded single-core figure, notebook, checker, and publication validation. No GPU was used.

No leaderboard score or point increase is claimed before live judge evaluation.
