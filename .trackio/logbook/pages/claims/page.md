# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_392d44e44479", "created_at": "2026-07-22T10:57:05+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Theorem 4.1 bounds the relative error between the true stationary covariance Σ_θ and the discrete-time proxy covariance Σ_ψ as ‖Σ_θ − Σ_ψ‖/‖Σ_θ‖ ≤ C_v λ^(1/2), scaling with the step size λ (Theorem 4.1, Section 4).
2. Theorem 4.3 derives an explicit formula for the minibatch noise covariance without assuming constant noise, enabling exact rather than approximate stationary covariance solutions (Theorem 4.3, Section 4).
3. Theorem 4.5 and Corollary 4.6 provide non-asymptotic 2-Wasserstein distance bounds between the true and proxy algorithm distributions, giving W₂(π_θ,π_ψ) ≤ Aλ/B for SGD (Theorem 4.5, Corollary 4.6).
4. Algorithm 1 gives a two-stage tuning procedure: offline sandwich covariance estimation from a subsample, followed by solving coupled equations for preconditioned step sizes (Algorithm 1).
5. On the Boston housing dataset under strong model misspecification, the proposed DQ+exact method maintains accurate covariance estimates while competing continuous-time and constant-noise methods diverge (Table 3).
6. Proposition B.1 extends the covariance and error-bound results to SGLD with momentum, recovering the non-momentum results as the momentum parameter κ→0 (Proposition B.1, Appendix B).
