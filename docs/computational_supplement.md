# Computational Supplement

This supplement records exploratory diagnostics, not independent prospective validation.

## Shorter Training Histories

Fractions are elapsed time within the original preterminal history, with a minimum of five points. Neural entries are 10-seed means; other entries are deterministic. Errors are mean curve MAE in % strain.

| Method | 50% time prefix | 75% time prefix |
|---|---:|---:|
| data_only | 0.320987 | 0.184833 |
| first_order | 0.276522 | 0.169207 |
| gompertz | 0.533583 | 0.161660 |
| larive | 0.293339 | 0.207805 |
| linear_last5 | 0.363677 | 0.322009 |
| persistence | 0.325374 | 0.177532 |
| pinn_shifted_first_order | 0.244310 | 0.127616 |
| shifted_avrami | 0.532154 | 0.189440 |
| shifted_first_order | 0.247408 | 0.213719 |

## Residual and Scaling Diagnostics

Raw derivative, right-hand-side, and residual RMS values are in `results/journal_check/review_residual_component_summary.csv`. Parameter ranges and the explicit 1%-of-support near-bound rule are in `review_head_bound_summary.csv`. Near-bound labels are not proof of exact numerical bound contact.

New five-seed domain, gate-aware sampling, and per-curve scaling results are in `review_new_sensitivity_metrics.csv`. The gate-aware sampling measure differs from uniform collocation. Per-curve scaling also changes dimensional head capacity support. These tests are not independent confirmations of the selected primary configuration.

## Conformal Diagnostic

For sorted calibration absolute errors $e_{(h)}$, the finite-sample index is $h=\lceil(n_{cal}+1)(1-\delta)\rceil$ and the half-width is $e_{(h)}$; if $h>n_{cal}$, use an infinite half-width. The 302 preterminal calibration residuals were reused for selection and are serially/source dependent. Empirical coverage therefore does not establish exchangeable or prospective coverage guarantees. The archived coverage files are computational diagnostics only.

## Remaining Scientific Checks

Equal optimizer-search budgets, onset-stratified rolling-origin evaluation, independent digitization error in strain units, and neutral treatment of ambiguous marker groups remain incomplete. The package must not be described as submission-ready.
