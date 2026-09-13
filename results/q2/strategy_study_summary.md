# Q2 Task 7C strategy study summary

The study uses one posterior sample and one final CandidateScore collection for all primary strategies and all rho values.

- Bayesian minimizes the nominal expected diameter proxy among candidates.
- Minimax minimizes the finite-grid robust proxy `u_proxy_m`; `u_bar_m` is only a conservative outer diagnostic.
- In this benchmark Bayesian has the smaller `psi_d_m` (43.451 m versus 44.936 m for minimax), while minimax has the smaller `u_proxy_m` (60.518 m versus 84.452 m for Bayesian).
- At rho=0.10 the hybrid exactly selects the minimax-side candidate. This is a valid outcome of the numerical envelope, not a failure, and rho=0.10 is neither a theoretical optimum nor a problem constant.
- The rho sweep is flat from 0 through 0.10, then switches candidates at 0.20 and 0.30: nominal `psi_d_m` decreases while `u_proxy_m` increases. The actual q/metrics are in `rho_sensitivity.csv`.
- Relative to the hybrid movement reference (1183.385 m), both same-distance vertical baselines have much larger `psi_d_m` (about 1206–1208 m) and `u_proxy_m` (1414.219 m).
- Both vertical sides are reported. Any equal or mirrored values are explained by the benchmark's x-axis symmetry.
- Values near `u_bar_m=1501.429` are conservative outer diagnostics, not true worst-case errors.

This is a single benchmark study, not a formal Monte Carlo claim or a strict global optimality result.
