# Simulation report: Brown-Resnick benchmark for seasonal extremal dependence

- Stations in panel: 25
- Pairs: 300
- Replications: 20
- Observations per season: 3150 (= 35 years x 90 days)
- Spectral truncation (n_factors): 50
- alpha: 1.0

## Regimes
- **null**: winter rho = 120.0, summer rho = 120.0
- **alternative**: winter rho = 180.0, summer rho = 60.0

## Approximation note
This run uses an **approximate** Brown-Resnick simulation via the spectral representation truncated at a finite number of Poisson points and with the Gaussian process anchored at the first station. The pairwise variance Var(W(s_i)-W(s_j)) = gamma(s_i, s_j) is preserved exactly, so the theoretical pairwise extremal coefficient theta(h) = 2 Phi(sqrt(gamma)/2) is preserved. Marginal distributions are only approximately unit Frechet; this is absorbed by the empirical rank transform used in the estimators.

## Bias and RMSE of theta

| scenario | season | theta_bias_mean | theta_rmse | n_pairs |
|---|---|---|---|---|
| alternative | summer | -0.0026 | 0.0117 | 6000 |
| alternative | winter | 0.0010 | 0.0059 | 6000 |
| null | summer | 0.0003 | 0.0082 | 6000 |
| null | winter | 0.0003 | 0.0075 | 6000 |

## Detection summary

Probability that the estimator correctly recovers the sign of the winter-summer difference, by distance bin. Under the alternative we expect proportions close to 1 (winter stronger); under the null we expect proportions near 0.5 (no systematic difference).

| scenario | dist_bin | correct_direction_theta | correct_direction_chi_0.90 | correct_direction_chi_0.95 | correct_direction_chi_0.98 |
|---|---|---|---|---|---|
| alternative | (-0.001, 50.0] | 1.000 | 1.000 | 1.000 | 0.966 |
| alternative | (50.0, 100.0] | 1.000 | 1.000 | 1.000 | 0.986 |
| alternative | (100.0, 150.0] | 1.000 | 1.000 | 1.000 | 0.994 |
| alternative | (150.0, 200.0] | 1.000 | 1.000 | 1.000 | 0.997 |
| alternative | (200.0, 10000.0] | 1.000 | 1.000 | 1.000 | 0.997 |
| null | (-0.001, 50.0] | 0.458 | 0.569 | 0.534 | 0.405 |
| null | (50.0, 100.0] | 0.466 | 0.623 | 0.535 | 0.405 |
| null | (100.0, 150.0] | 0.479 | 0.637 | 0.575 | 0.398 |
| null | (150.0, 200.0] | 0.503 | 0.629 | 0.577 | 0.387 |
| null | (200.0, 10000.0] | 0.562 | 0.663 | 0.565 | 0.388 |

## Distance-bin summary (theta_hat)

| scenario | season | dist_bin | theta_hat_mean | theta_true_mean | theta_hat_std |
|---|---|---|---|---|---|
| alternative | summer | (-0.001, 50.0] | 1.2925 | 1.2934 | 0.0437 |
| alternative | summer | (50.0, 100.0] | 1.4190 | 1.4208 | 0.0387 |
| alternative | summer | (100.0, 150.0] | 1.5206 | 1.5231 | 0.0289 |
| alternative | summer | (150.0, 200.0] | 1.5981 | 1.6016 | 0.0232 |
| alternative | summer | (200.0, 10000.0] | 1.6724 | 1.6771 | 0.0301 |
| alternative | winter | (-0.001, 50.0] | 1.1729 | 1.1722 | 0.0261 |
| alternative | winter | (50.0, 100.0] | 1.2525 | 1.2514 | 0.0246 |
| alternative | winter | (100.0, 150.0] | 1.3200 | 1.3189 | 0.0190 |
| alternative | winter | (150.0, 200.0] | 1.3754 | 1.3743 | 0.0164 |
| alternative | winter | (200.0, 10000.0] | 1.4328 | 1.4323 | 0.0238 |
| null | summer | (-0.001, 50.0] | 1.2097 | 1.2100 | 0.0315 |
| null | summer | (50.0, 100.0] | 1.3050 | 1.3053 | 0.0297 |
| null | summer | (100.0, 150.0] | 1.3853 | 1.3852 | 0.0230 |
| null | summer | (150.0, 200.0] | 1.4507 | 1.4498 | 0.0201 |
| null | summer | (200.0, 10000.0] | 1.5174 | 1.5158 | 0.0290 |
| null | winter | (-0.001, 50.0] | 1.2104 | 1.2100 | 0.0319 |
| null | winter | (50.0, 100.0] | 1.3056 | 1.3053 | 0.0295 |
| null | winter | (100.0, 150.0] | 1.3856 | 1.3852 | 0.0225 |
| null | winter | (150.0, 200.0] | 1.4503 | 1.4498 | 0.0188 |
| null | winter | (200.0, 10000.0] | 1.5154 | 1.5158 | 0.0271 |

## Methodological interpretation

The simulation tests whether the pairwise F-madogram, theta, and finite-level chi_u estimators can recover the true spatial extremal dependence in finite samples comparable to the empirical KNMI panel. Under the alternative, the simulation generates winter fields with a larger range parameter than summer, so the true theta_winter(h) is strictly below theta_summer(h) for relevant distances. A high detection rate under the alternative, together with a near-0.5 detection rate under the null, supports the use of these estimators in the empirical analysis.
