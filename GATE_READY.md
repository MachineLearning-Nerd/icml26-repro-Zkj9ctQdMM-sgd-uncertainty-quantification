# Release gate — Zkj9ctQdMM

**Scoped evidence gate: PASSED. Paper-level assessment: INCONCLUSIVE.**

- Claim 5 is **FALSIFIED as registered**: the paper’s own Table 3 and the full-Boston replay show finite CT covariance error.
- Full Boston housing data: 506 rows, 13 features plus intercept.
- Formal compute: Hugging Face `cpu-upgrade`; no GPU.
- Raw evidence: `outputs/c5_falsification.json`.
- Deterministic falsification gate: `repro/src/check_claim5_falsification.py`.
- Eight of eight Claim 5 checks pass in the evidence-bearing run.
- Claims 1–4 and 6 remain resolved in the cumulative suite, with the conditional/scoped limits documented in `README.md` and `STATUS.md`.
- The release-validation child passed in 17m52s with exit 0.
- Fresh-clone checker, deterministic figure regeneration, artifact integrity, Python compilation, and strict marimo validation pass.
- The conservative machine-readable assessment is `publication_gate.json`.
