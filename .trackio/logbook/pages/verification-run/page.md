# Verification run (current — supersedes the historical toy run)

**Current verifier:** `repro/src/verify_sgduq.py` on branch `orx/claim5-release-validation` @ `94def38`.
This replaces the [historical rejected baseline](#/historical-rejected-baseline/verification-run) (toy 2D verifier, judged 2/12).

## Release validation — `cpu-upgrade`, exit 0, 17m52s (current)

Command (fixed contract): `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py`

This immutable child reran the evidence-bearing source state after the canonical Claim 5 page and raw JSON were committed. It reproduced all eight Claim 5 PASS conditions, kept claims 1–4 and 6 VERIFIED, and exited 0. No GPU was used.

## Evidence-generating run — `cpu-upgrade`, exit 0, 19 min
Command (fixed contract): `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py`

Claims 1–4 and 6 re-ran bit-identically (same seeds) and remained **VERIFIED**. Claim 5 executed the full falsification campaign (`claim5_falsify.py`, seed 20260729):

```
--- variant icpt-post (PRIMARY): full Boston 506x13, target S*/N, 30 chains/cell
 B=16 DQ+exact  rho=0.9057 exact_err=0.0152 sim:30fin/0div med=0.0405 -> finite
 B=16 DQ+const  rho=3.1562 exact_err=inf    sim:0fin/30div            -> DIVERGES
 B=16 LR+WS     rho=1.5639 exact_err=inf    sim med=6.26e+04          -> DIVERGES
 B=16 CT        rho=0.8964 exact_err=0.0730 sim:30fin/0div med=0.0768 -> finite
 B=50 DQ+exact  rho=0.7375 exact_err=0.0152 sim:30fin/0div med=0.0240 -> finite
 B=50 DQ+const  rho=2.9181 exact_err=inf    sim:0fin/30div            -> DIVERGES
 B=50 LR+WS     rho=1.7073 exact_err=inf    sim med=1.96e+04          -> DIVERGES
 B=50 CT        rho=0.7028 exact_err=0.1548 sim:30fin/0div med=0.1552 -> finite
 (controls: CTx50 diverges rho=52/226; wellspec-gauss makes LR+WS accurate 0.081;
  without-replacement: same pattern; ceny/stdy variants: CT finite everywhere)
  [PASS] A_regime_reproduced  [PASS] B_ct_does_not_diverge  [PASS] C_dq_exact_accurate
  [PASS] D_lrws_diverges      [PASS] E_detector_control     [PASS] F_wellspec_control
  [PASS] G_crosscheck         [PASS] H_ct_robust_across_variants
  -> FALSIFIED
== VERDICT SUMMARY ==
  [VERIFIED] c1 c2 c3 c4 c6   [FALSIFIED] c5_boston_table3
  6/6 claims resolved (VERIFIED or FALSIFIED)  (elapsed 1124s) -- exit 0
```
The run log also embeds the raw `outputs/c5_falsification.json` verbatim between `===BEGIN/END c5_falsification.json===` markers (evidence provenance).

## HF run `c626a479` — `cpu-upgrade`, exit 0, 33 s
Command (fixed contract): `pip install --quiet numpy==2.5.1 scipy==1.18.0 sympy==1.14.0 && python repro/src/verify_sgduq.py`

```
Reproduction: Accurate Large-sample UQ using SG-MCMC (arXiv 2606.00293)
numpy 2.5.1; seed base 20260724
== CLAIM 1 (Theorem 4.1): relative covariance error = O(sqrt(lambda)) ==
  fitted rel_err ~ lam^-0.480 (MC-noise-limited; theorem rate 0.5 derived symbolically)
  exact Sigma_psi matches proxy-chain MC: True; control exponent -0.113
  -> VERIFIED  (sym=True exact=True Cv_finite=True control=True)
== CLAIM 2 (Theorem 4.3): exact minibatch noise covariance Cbar_psi (Eq 16) ==
  Eq16 vs MC rel=0.017/0.022/0.032 ; differs from const-noise by 0.046/0.046/0.078
  -> VERIFIED  (sym=True mc=True exact!=const=True)
== CLAIM 3 (Theorem 4.5/Cor 4.6): W2(pi_theta,pi_psi) = O(lambda/B) ==
  bound W2<=A*lam/B holds with A=1.56; control W2(mis-centered)=0.0585 > W2(correct)=0.0058
  -> VERIFIED  (bound=True control=True)
== CLAIM 4 (Algorithm 1): two-stage DQ+exact tuning ==
  DQ+exact rel-to-target = 0.324; chain empirical rel-to-theory = 0.084
  -> VERIFIED
== CLAIM 5 (Table 3): Boston housing ==
  B=16 (D=6 Boston subset): DQ+exact=0.2069  DQ+const=0.6551  LR+WS=0.3701  CT=0.9915
  -> VERIFIED
== CLAIM 6 (Proposition B.1): momentum; kappa->0 ==
  kappa=0.000 rel_err=4.6e-16
  -> VERIFIED  (sym=True limit=True)
== VERDICT SUMMARY ==
  [VERIFIED] c1_thm4.1_sqrt_lambda   [VERIFIED] c2_thm4.3_exact_noise
  [VERIFIED] c3_thm4.5_W2_bound      [VERIFIED] c4_algorithm1_two_stage
  [VERIFIED] c5_boston_table3        [VERIFIED] c6_propB1_momentum
  6/6 claims VERIFIED  (elapsed 33s)
```
(Full per-claim raw metrics in `outputs/c{1..6}_*.json`.) The verifier exits **nonzero** if any VERIFIED-required check fails; this run exited 0.
