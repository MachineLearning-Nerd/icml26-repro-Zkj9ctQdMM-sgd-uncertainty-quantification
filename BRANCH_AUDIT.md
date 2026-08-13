# Branch audit

This file records what each source branch contributed before branch cleanup. The old names are historical experiment labels, not supported entry points. Commit IDs below are the source refs observed before the final identity normalization; the publication cleanup may rewrite their hashes. The release artifacts and role summary are preserved on the canonical `main` surface; branch-only commits may become unreachable after the stale remote refs are removed.

## Source branch map

| Source ref | Tip observed before cleanup | Role | Evidence or limitation |
|---|---|---|---|
| `orx/baseline` | `923a9df` | Added the pinned UV environment and Boston data | Historical scaffold; early toy baseline, not release evidence |
| `orx/faithful-symbolic-numeric-repro` | `242eac2` | Added clean-room symbolic Eq. 15/Eq. 16/Prop. B.1 checks, exact discrete-time numerics, logistic and initial Boston work | Mechanism development; Claim 5 was not yet full-Boston release evidence |
| `orx/faithful-pip-repro` | `d6bb15c` | Pinned pip reproduction of the cumulative six-claim suite and Trackio evidence pages | Claim 5 used a six-feature toy; its `VERIFIED` result is superseded by the full-Boston audit |
| `orx/claim5-falsification-full-boston` | `94def38` | Full 506×13 Boston replay, exact second-moment operator, controls, raw JSON, and deterministic checker | Produced the evidence that CT remains finite while LR+WS/DQ+const are unstable |
| `orx/claim5-release-validation` | `94def38` | Immutable release rerun of the evidence-bearing state | 8/8 Claim 5 checks passed; Claims 1–4 and 6 remained resolved |
| `release-claim5-falsification` | `93f43f7` | Clean-clone publication-gate handoff | Validated the release surface and merged the handoff |
| `master` / `main` | `093af97` | Reader-facing publication surface | Same pre-cleanup tip; `main` becomes the only canonical branch |

## Cleanup policy

1. Preserve the branch roles and source refs here.
2. Rename the repository to `icml26-sgd-uncertainty-quantification`.
3. Rename the canonical branch to `main`.
4. Remove stale `master`, `orx/*`, and `release-*` remote branches after the final evidence and docs are reachable from `main`.
5. Normalize reachable commit author and committer identities to the `MachineLearning-Nerd` GitHub identity.

The branch cleanup is a namespace cleanup of the supported GitHub entry points. It does not rewrite the committed raw evidence, reports, or historical Trackio pages that remain on `main`; branch-only commits are represented here by their source ref and role.
