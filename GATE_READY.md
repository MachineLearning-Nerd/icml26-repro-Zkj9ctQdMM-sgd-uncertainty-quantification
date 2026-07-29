# Release gate — Zkj9ctQdMM

Claim 5 status: **FALSIFIED as registered**.

- Full Boston housing data: 506 rows, 13 features plus intercept.
- Formal compute: Hugging Face `cpu-upgrade`; no GPU.
- Raw evidence: `outputs/c5_falsification.json`.
- Deterministic falsification gate: `repro/src/check_claim5_falsification.py`.
- Eight of eight Claim 5 checks pass in the evidence-bearing run.
- Claims 1–4 and 6 remain unchanged in the cumulative suite.
- The `claim5-release-validation` child passed in 17m52s with exit 0.
- Publication remains gated only on the fresh-clone checks.
