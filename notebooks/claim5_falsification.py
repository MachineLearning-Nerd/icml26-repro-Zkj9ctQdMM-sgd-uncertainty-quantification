import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Boston housing Claim 5: the CT method stays finite

    **Evidence first.** The registered claim says DQ+exact is accurate while
    continuous-time (CT) and constant-noise competitors diverge. The paper's
    own Table 3 and an independent full-data replay both show finite CT:

    | Evidence | Batch | DQ+exact | CT | LR+WS |
    |---|---:|---:|---:|---:|
    | Paper Table 3 | 16 | 0.337 | **0.247** | 9.23×10⁸ |
    | Paper Table 3 | 50 | 0.352 | **0.589** | 1.40×10⁷ |
    | Replay, simulated median | 16 | 0.041 | **0.077** | 6.26×10⁴ |
    | Replay, simulated median | 50 | 0.024 | **0.155** | 1.96×10⁴ |

    **Status: FALSIFIED as registered.** CT is finite at both batches while
    the LR+WS instability is simultaneously reproduced.
    """)
    return


@app.cell
def _():
    evidence = {
        16: {
            "DQ+exact": {"rho": 0.905686, "exact_error": 0.015250, "median": 0.040539},
            "CT": {"rho": 0.896410, "exact_error": 0.072965, "median": 0.076842},
            "LR+WS": {"rho": 1.563931, "exact_error": float("inf"), "median": 62603.860102},
            "DQ+const": {"rho": 3.156155, "exact_error": float("inf"), "median": float("inf")},
        },
        50: {
            "DQ+exact": {"rho": 0.737489, "exact_error": 0.015250, "median": 0.024002},
            "CT": {"rho": 0.702840, "exact_error": 0.154762, "median": 0.155168},
            "LR+WS": {"rho": 1.707344, "exact_error": float("inf"), "median": 19580.708751},
            "DQ+const": {"rho": 2.918081, "exact_error": float("inf"), "median": float("inf")},
        },
    }
    return (evidence,)


@app.cell
def _(mo):
    batch = mo.ui.dropdown(options=[16, 50], value=16, label="Inspect batch size")
    batch
    return (batch,)


@app.cell
def _(batch, evidence, mo):
    rows = []
    for method, values in evidence[batch.value].items():
        status = "stable" if values["rho"] < 1 else "diverges"
        exact = "∞" if values["exact_error"] == float("inf") else f'{values["exact_error"]:.3f}'
        median = "∞" if values["median"] == float("inf") else f'{values["median"]:.3g}'
        rows.append(
            f"| {method} | {values['rho']:.3f} | {status} | {exact} | {median} |"
        )
    mo.md(
        "\n".join(
            [
                f"## Exact and empirical cross-check at B={batch.value}",
                "",
                "| Method | ρ(T) | Exact status | Exact error | Simulated median |",
                "|---|---:|---|---:|---:|",
                *rows,
                "",
                r"For quadratic log loss, the exact minibatch chain has a finite "
                r"stationary second moment if and only if $\rho(T)<1$.",
            ]
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the counterexample is in-domain

    The replay uses all 506 Boston rows, 13 standardized features plus an
    intercept, log loss, the paper's two batch sizes, and the paper's relative
    Frobenius covariance-error metric. The target is the posterior-scale
    sandwich covariance \(S^\star/N\).

    Boston's OLS residual variance is **21.895**, while LR+WS uses the
    well-specified model value \(\sigma^2=1\). In a Gaussian unit-noise control,
    residual variance is 1.047 and LR+WS returns to \(\rho=0.893\) with exact
    error 0.081. The same implementation therefore behaves correctly when its
    assumptions hold.

    A detector control multiplies CT's step matrix by 50 and is correctly
    identified as divergent (\(\rho=51.9\) and 226.5). Finiteness is not a
    failure of the detector.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproducibility contract

    Formal command:

    ```bash
    pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py
    ```

    The formal run used Hugging Face `cpu-upgrade`, seed `20260729`, 30
    chains per cell, 15,000 burn-in iterations, and 60,000 measured iterations.
    No GPU was used.

    This notebook embeds the already-produced headline evidence. It does not
    launch the expensive experiment. See the repository report, raw JSON, and
    deterministic checker for the complete audit trail.
    """)
    return


if __name__ == "__main__":
    app.run()
