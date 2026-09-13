# Q2 rho sensitivity

All rows reuse the same final CandidateScore collection; only the hybrid selector is called again.

The dashed reference in the companion plot marks rho=0.10 for comparison; it is a modeling/sensitivity reference, not a theoretically optimal rho. Values are flat from rho=0 to 0.10 (minimax plateau), then switch candidates at rho=0.20 and 0.30. Relative to that plateau, nominal psi improves as robustness is relaxed; discrete candidates and tau_m=0.5 near-optimal tie-breaking do not require local monotonicity of reported psi.

| rho | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | Δpsi vs minimax | Δu_proxy vs minimax |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.000 | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | 0.000 | 0.000 |
| 0.025 | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | 0.000 | 0.000 |
| 0.050 | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | 0.000 | 0.000 |
| 0.100 | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | 0.000 | 0.000 |
| 0.200 | (28.28,-571.72) | 42.985 | 69.914 | 1501.429 | 1090.216 | 1.952 | 9.396 |
| 0.300 | (-0.00,560.00) | 43.254 | 73.171 | 1501.429 | 1060.000 | 1.682 | 12.654 |
