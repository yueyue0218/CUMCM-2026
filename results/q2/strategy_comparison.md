# Q2 Task 7C strategy comparison

One shared final score set; runtime=165.670 s; prior_draws=600000, retained=1611, ESS=1218.692.

| strategy | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | C_rec |
|---|---|---:|---:|---:|---:|---|
| pure_bayesian | (0.00,-480.00) | 43.451 | 84.452 | 1501.429 | 1020.000 | False |
| pure_minimax | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| hybrid_rho_0.10 | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| vertical_plus | (-900.00,1183.38) | 1206.091 | 1414.219 | 1501.429 | 1183.385 | False |
| vertical_minus | (-900.00,-1183.38) | 1207.615 | 1414.219 | 1501.429 | 1183.385 | False |

The pure Bayesian selector accepts candidates within tau_m=0.5 m of the best nominal psi and then applies the existing secondary tie-break. Vertical rows are both retained; they use the rho=0.10 hybrid movement as reference. Under this symmetric benchmark, equal or mirrored values reflect x-axis symmetry rather than a physical preference.

Relative to pure minimax, the active strategy rows have psi/u_proxy differences (positive means larger):
- `pure_bayesian`: Δpsi=-1.485 m; Δu_proxy=23.934 m
- `pure_minimax`: Δpsi=0.000 m; Δu_proxy=0.000 m
- `hybrid_rho_0.10`: Δpsi=0.000 m; Δu_proxy=0.000 m
