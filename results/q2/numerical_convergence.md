# Q2 emergency compact discretization sanity check

Fixed prior_draws=600000; refinement=(120,40), circle_vertices=72, rho=0.10, tau_m=0.5.

| part | parameter | value | strategy | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | C_rec |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| B | direction_bins | 180 | pure_bayesian | (0.00,-480.00) | 52.396 | 84.452 | 1501.429 | 1020.000 | False |
| B | direction_bins | 180 | pure_minimax | (120.00,-600.00) | 57.469 | 58.324 | 1501.429 | 1183.385 | False |
| B | direction_bins | 180 | hybrid | (28.28,-571.72) | 54.418 | 63.120 | 1501.429 | 1090.216 | False |
| B | direction_bins | 360 | pure_bayesian | (0.00,-480.00) | 43.451 | 84.452 | 1501.429 | 1020.000 | False |
| B | direction_bins | 360 | pure_minimax | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| B | direction_bins | 360 | hybrid | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| C | coarse_spacing_m | 300.0 | pure_bayesian | (0.00,-480.00) | 43.451 | 84.452 | 1501.429 | 1020.000 | False |
| C | coarse_spacing_m | 300.0 | pure_minimax | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| C | coarse_spacing_m | 300.0 | hybrid | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| C | coarse_spacing_m | 200.0 | pure_bayesian | (0.00,-480.00) | 43.451 | 84.452 | 1501.429 | 1020.000 | False |
| C | coarse_spacing_m | 200.0 | pure_minimax | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |
| C | coarse_spacing_m | 200.0 | hybrid | (120.00,-600.00) | 44.936 | 60.518 | 1501.429 | 1183.385 | False |

Comparisons use the explicit x-axis symmetry-aware distance
`min(||q_old-q_new||, ||q_old-reflect_x(q_new)||)` because this benchmark has
arena center `(0,0)`, station `(-900,0)`, first bearing `0°`, and a nominal
error model symmetric about zero. This is an engineering numerical check, not
a strict convergence theorem; a non-symmetric benchmark must use ordinary
Euclidean displacement.

Direction 180→360: Bayesian Δq_sym=0 m, minimax Δq_sym=0 m, hybrid
Δq_sym≈30.615 m. Positions and robust proxies are broadly stable, but
`psi_d_m` changes materially: Bayesian 52.396→43.451 (-17.1%), minimax
57.469→44.936 (-21.8%), and hybrid 54.418→44.936 (-17.4%). No 720-bin check
was run because of time/compute cost. Spatial 300→200 gives identical selected
points and metrics for all three strategies in this run (Δq_sym=0 m).
