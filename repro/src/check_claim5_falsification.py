"""Deterministic verifier for the Claim 5 falsification (Table 3, Boston housing).

Reads outputs/c5_falsification.json (produced by claim5_falsify.py) and checks every
condition the falsification requires. EXITS NONZERO when the claimed falsification is
absent, so this script is a machine-checkable gate:

  A. regime      -- a pre-registered primary variant exists: DQ+exact accurate AND
                    LR+WS divergent at both batch sizes (the claim's own premises).
  B. ct-finite   -- in the primary variant CT does NOT diverge at either batch size:
                    rho < 1, exact stationary error finite and < 10, all simulated
                    chains finite with median error < 10. This contradicts the
                    registered claim's "continuous-time ... methods diverge".
  C. dq-accurate -- DQ+exact: rho < 1, exact err < 1, all chains finite, median < 1.
  D. lrws-diverges  -- LR+WS diverged at both batch sizes (positive control: the
                    divergence phenomenon the claim describes is really present).
  E. detector    -- CTx50 negative control diverges by BOTH routes at both batch
                    sizes (finiteness findings are not a detector failure).
  F. wellspec    -- Gaussian-covariate well-specified control: LR+WS is stable and
                    accurate (rho < 1, exact err < 0.5, all chains finite), so the
                    Boston divergence is caused by the misspecification the claim
                    conditions on, not by our LR+WS implementation.
  G. crosscheck  -- DQ+exact-asym (closed-form exact solution of eq (11)) reproduces
                    the target to < 1% in the primary variant, and for CT and DQ+exact
                    the simulated median agrees with the exact stationary error
                    (|median - exact| < max(0.5, 2x exact)): the two independent
                    evidence routes are mutually consistent.
  H. robustness  -- CT is finite (rho < 1 and exact err < 10) at both batch sizes in
                    EVERY posterior-scale variant and in the without-replacement
                    sensitivity: the counterexample does not depend on protocol
                    choices the paper leaves open.

Usage: uv run --locked python repro/src/check_claim5_falsification.py
"""
from __future__ import annotations
import json
import math
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def evaluate(out):
    checks = {}
    B_list = [str(b) for b in out["B_list"]]
    primary = out.get("primary")

    checks["A_regime_reproduced"] = primary is not None
    if primary is None:
        return False, checks

    cells = primary["cells"]

    def cell(B, m):
        return cells[B][m]

    checks["B_ct_does_not_diverge"] = all(
        cell(B, "CT")["rho"] < 1.0
        and finite(cell(B, "CT")["exact_cov_err"])
        and cell(B, "CT")["exact_cov_err"] < out["finite_thresh"]
        and cell(B, "CT")["sim"]["diverged_runs"] == 0
        and finite(cell(B, "CT")["sim"]["median"])
        and cell(B, "CT")["sim"]["median"] < out["finite_thresh"]
        for B in B_list)

    checks["C_dq_exact_accurate"] = all(
        cell(B, "DQ+exact")["rho"] < 1.0
        and finite(cell(B, "DQ+exact")["exact_cov_err"])
        and cell(B, "DQ+exact")["exact_cov_err"] < 1.0
        and cell(B, "DQ+exact")["sim"]["diverged_runs"] == 0
        and cell(B, "DQ+exact")["sim"]["median"] < 1.0
        for B in B_list)

    checks["D_lrws_diverges"] = all(cell(B, "LR+WS")["diverged"] for B in B_list)

    checks["E_detector_control"] = all(
        cell(B, "CTx50-control")["rho"] >= 1.0
        and cell(B, "CTx50-control")["sim"]["diverged_runs"] > 0
        for B in B_list)

    ws = out.get("controls", {}).get("wellspec_gauss")
    checks["F_wellspec_control"] = ws is not None and all(
        ws["cells"][B]["LR+WS"]["rho"] < 1.0
        and finite(ws["cells"][B]["LR+WS"]["exact_cov_err"])
        and ws["cells"][B]["LR+WS"]["exact_cov_err"] < 0.5
        and ws["cells"][B]["LR+WS"]["sim"]["diverged_runs"] == 0
        for B in B_list)

    def sim_exact_agree(c):
        e, m = c["exact_cov_err"], c["sim"]["median"]
        return finite(e) and finite(m) and abs(m - e) < max(0.5, 2.0 * e)

    checks["G_crosscheck"] = all(
        cell(B, "DQ+exact-asym")["exact_cov_err"] < 0.01
        and sim_exact_agree(cell(B, "CT"))
        and sim_exact_agree(cell(B, "DQ+exact"))
        for B in B_list)

    wor = out.get("controls", {}).get("without_replacement")
    post_variants = [v for v in out["variants"] if v["nscale"]]
    if wor is not None:
        post_variants = post_variants + [wor]
    checks["H_ct_robust_across_variants"] = all(
        v["cells"][B]["CT"]["rho"] < 1.0
        and finite(v["cells"][B]["CT"]["exact_cov_err"])
        and v["cells"][B]["CT"]["exact_cov_err"] < out["finite_thresh"]
        for v in post_variants for B in B_list)

    return all(checks.values()), checks


def main():
    path = os.path.join(OUT, "c5_falsification.json")
    if not os.path.exists(path):
        print(f"MISSING: {path} -- run claim5_falsify.py first")
        sys.exit(3)
    out = json.load(open(path))
    ok, checks = evaluate(out)
    print("Claim 5 falsification verifier")
    print(f"  primary variant: {out.get('primary_tag')}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if ok:
        p = out["primary"]["cells"]
        print("\n  FALSIFICATION PRESENT: in the reproduced Table-3 regime the",
              "continuous-time (CT) tuning does not diverge",
              f"(B=16: rho={p['16']['CT']['rho']:.3f}, err={p['16']['CT']['exact_cov_err']:.3f};",
              f"B=50: rho={p['50']['CT']['rho']:.3f}, err={p['50']['CT']['exact_cov_err']:.3f})",
              "while LR+WS diverges -- contradicting the registered claim's",
              "'continuous-time and constant-noise methods diverge'.")
        print("  Paper's own Table 3 concurs: CT = 0.247 / 0.589 (finite), LR+WS = 9.23e8 / 1.40e7.")
    else:
        print("\n  FALSIFICATION ABSENT under this run; the registered claim verdict is preserved.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
