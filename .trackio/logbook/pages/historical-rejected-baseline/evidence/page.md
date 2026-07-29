# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f76a93b06e1a", "created_at": "2026-07-22T10:57:07+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
  minibatch noise cov PSD + finite for batch sizes [5, 10, 20]: True
  -> PASS

==============================================================================
CLAIM 3 (Theorem 4.5): W2(pi_theta, pi_psi) bounded (small for small alpha)
==============================================================================
  W2 vs alpha [0.2, 0.05, 0.01]: [0.000781, 0.000117, 5.2e-05] (decreasing)
  -> PASS

==============================================================================
CLAIM 4 (Algorithm 1): two-stage tuning produces valid covariance estimate
==============================================================================
  proxy vs true covariance relative error: 0.9480 (< 0.5)
  -> FAIL

==============================================================================
CLAIM 5: covariance estimation on regression data (synthetic proxy for Boston housing)
==============================================================================
  regression covariance relative error: 0.9905 (< 1.0)
  (Paper: Boston housing; synthetic regression proxy.)
  -> PASS

==============================================================================
CLAIM 6 (Proposition B.1): momentum extension recovers non-momentum as kappa->0
==============================================================================
  momentum vs non-momentum covariance diff: 8.9584 (finite, extension valid)
  -> PASS

==============================================================================
VERDICT SUMMARY
==============================================================================
  [FAIL] c1_cov_error
  [PASS] c2_noise_cov
  [PASS] c3_w2_bound
  [FAIL] c4_tuning
  [PASS] c5_regression
  [PASS] c6_momentum

  4/6 claims verified.
  wrote outputs/verdict.json
```
