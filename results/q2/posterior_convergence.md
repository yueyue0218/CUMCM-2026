# Q2 Task 7B-1: posterior convergence and strategy stability

Fixed configuration: station `(-900,0)`, first bearing `0°`, seed `7`, coarse spacing `300 m`, refinement `(120,40) m`, direction grid `180`, circle vertices `72`, `rho=0.10`, `tau_m=0.5`. The nominal error model is the explicit experiment model reported by `src/q2/main.py`.

The main robust-envelope hybrid uses `u_proxy_m`; `u_bar_m` is reported only as a conservative outer diagnostic.

## Posterior diagnostics

| prior_draws | runtime (s) | retained | ESS | acceptance |
|---:|---:|---:|---:|---:|
| 10000 | 8.914 | 41 | 30.752 | 0.004100 |
| 30000 | 10.967 | 84 | 63.810 | 0.002800 |
| 60000 | 15.165 | 158 | 119.753 | 0.002633 |
| 100000 | 18.211 | 256 | 194.963 | 0.002560 |
| 300000 | 57.981 | 820 | 622.710 | 0.002733 |
| 600000 | 124.101 | 1611 | 1218.692 | 0.002685 |
| 1000000 | 156.886 | 2741 | 2076.570 | 0.002741 |

## Strategy stability

| prior_draws | strategy | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | C_rec | Δq from previous |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 10000 | pure_bayesian | (0.00,300.00) | 14.414 | 53.041 | 165.313 | 948.683 | True | — |
| 10000 | pure_minimax | (-84.85,-515.15) | 19.358 | 32.096 | 148.068 | 964.283 | True | — |
| 10000 | hybrid | (-120.00,300.00) | 15.899 | 34.077 | 204.721 | 835.703 | True | — |
| 30000 | pure_bayesian | (-120.00,300.00) | 24.753 | 58.353 | 204.721 | 835.703 | True | 120.000 |
| 30000 | pure_minimax | (-0.00,480.00) | 25.500 | 44.519 | 1501.429 | 1020.000 | False | 998.758 |
| 30000 | hybrid | (-0.00,480.00) | 25.500 | 44.519 | 1501.429 | 1020.000 | False | 216.333 |
| 60000 | pure_bayesian | (-40.00,300.00) | 31.963 | 70.271 | 175.691 | 910.824 | True | 80.000 |
| 60000 | pure_minimax | (-28.28,571.72) | 35.539 | 44.519 | 1501.429 | 1042.472 | False | 95.978 |
| 60000 | hybrid | (-0.00,480.00) | 33.646 | 47.185 | 1501.429 | 1020.000 | False | 0.000 |
| 100000 | pure_bayesian | (-0.00,480.00) | 38.617 | 47.185 | 1501.429 | 1020.000 | False | 184.391 |
| 100000 | pure_minimax | (-0.00,560.00) | 39.723 | 46.466 | 1501.429 | 1060.000 | False | 30.615 |
| 100000 | hybrid | (-0.00,480.00) | 38.617 | 47.185 | 1501.429 | 1020.000 | False | 0.000 |
| 300000 | pure_bayesian | (0.00,-480.00) | 48.093 | 68.749 | 1501.429 | 1020.000 | False | 960.000 |
| 300000 | pure_minimax | (120.00,-600.00) | 54.278 | 57.519 | 1501.429 | 1183.385 | False | 1166.190 |
| 300000 | hybrid | (0.00,-560.00) | 50.434 | 62.352 | 1501.429 | 1060.000 | False | 1040.000 |
| 600000 | pure_bayesian | (0.00,-480.00) | 52.396 | 84.452 | 1501.429 | 1020.000 | False | 0.000 |
| 600000 | pure_minimax | (120.00,-600.00) | 57.469 | 58.324 | 1501.429 | 1183.385 | False | 0.000 |
| 600000 | hybrid | (28.28,-571.72) | 54.418 | 63.120 | 1501.429 | 1090.216 | False | 30.615 |
| 1000000 | pure_bayesian | (-0.00,480.00) | 55.146 | 82.435 | 1501.429 | 1020.000 | False | 960.000 |
| 1000000 | pure_minimax | (120.00,-600.00) | 59.346 | 62.585 | 1501.429 | 1183.385 | False | 0.000 |
| 1000000 | hybrid | (-0.00,560.00) | 56.284 | 65.975 | 1501.429 | 1060.000 | False | 1132.069 |

## Adjacent-size changes

Relative changes use `max(|previous|, 1e-12)` as denominator; absolute changes are retained in CSV.

- `pure_bayesian` at 30000: Δq=120.000 m; Δpsi=10.340 m (0.717357); Δu_proxy=5.313 m (0.100164); Δu_bar=39.408 m (0.238383)
- `pure_minimax` at 30000: Δq=998.758 m; Δpsi=6.143 m (0.317313); Δu_proxy=12.423 m (0.387056); Δu_bar=1353.361 m (9.14011)
- `hybrid` at 30000: Δq=216.333 m; Δpsi=9.602 m (0.603916); Δu_proxy=10.442 m (0.306423); Δu_bar=1296.708 m (6.33404)
- `pure_bayesian` at 60000: Δq=80.000 m; Δpsi=7.210 m (0.291272); Δu_proxy=11.918 m (0.20424); Δu_bar=29.029 m (-0.141799)
- `pure_minimax` at 60000: Δq=95.978 m; Δpsi=10.039 m (0.393673); Δu_proxy=0.000 m (0); Δu_bar=0.000 m (0)
- `hybrid` at 60000: Δq=0.000 m; Δpsi=8.145 m (0.319422); Δu_proxy=2.667 m (0.0598964); Δu_bar=0.000 m (0)
- `pure_bayesian` at 100000: Δq=184.391 m; Δpsi=6.654 m (0.20818); Δu_proxy=23.086 m (-0.328526); Δu_bar=1325.738 m (7.54583)
- `pure_minimax` at 100000: Δq=30.615 m; Δpsi=4.183 m (0.117712); Δu_proxy=1.947 m (0.0437278); Δu_bar=0.000 m (0)
- `hybrid` at 100000: Δq=0.000 m; Δpsi=4.971 m (0.147755); Δu_proxy=0.000 m (0); Δu_bar=0.000 m (0)
- `pure_bayesian` at 300000: Δq=960.000 m; Δpsi=9.476 m (0.245375); Δu_proxy=21.564 m (0.457005); Δu_bar=0.000 m (0)
- `pure_minimax` at 300000: Δq=1166.190 m; Δpsi=14.556 m (0.366437); Δu_proxy=11.053 m (0.237883); Δu_bar=0.000 m (0)
- `hybrid` at 300000: Δq=1040.000 m; Δpsi=11.817 m (0.306003); Δu_proxy=15.166 m (0.321422); Δu_bar=0.000 m (0)
- `pure_bayesian` at 600000: Δq=0.000 m; Δpsi=4.303 m (0.0894822); Δu_proxy=15.702 m (0.228396); Δu_bar=0.000 m (0)
- `pure_minimax` at 600000: Δq=0.000 m; Δpsi=3.191 m (0.058788); Δu_proxy=0.805 m (0.013999); Δu_bar=0.000 m (0)
- `hybrid` at 600000: Δq=30.615 m; Δpsi=3.984 m (0.0789855); Δu_proxy=0.768 m (0.0123117); Δu_bar=0.000 m (0)
- `pure_bayesian` at 1000000: Δq=960.000 m; Δpsi=2.749 m (0.0524751); Δu_proxy=2.017 m (-0.0238802); Δu_bar=0.000 m (0)
- `pure_minimax` at 1000000: Δq=0.000 m; Δpsi=1.877 m (0.0326633); Δu_proxy=4.261 m (0.0730523); Δu_bar=0.000 m (0)
- `hybrid` at 1000000: Δq=1132.069 m; Δpsi=1.867 m (0.0342999); Δu_proxy=2.855 m (0.0452369); Δu_bar=0.000 m (0)

## Interpretation and risks

Observed 600,000→1,000,000: pure_bayesian Δq=960.000 m, Δpsi=0.0525, Δu_proxy=-0.0239; pure_minimax Δq=0.000 m, Δpsi=0.0327, Δu_proxy=0.0731; hybrid Δq=1132.069 m, Δpsi=0.0343, Δu_proxy=0.0452.
The selections therefore show partial stability but not blanket convergence; judge each strategy using ESS and the reported adjacent changes.

These finite-sample results are a stability diagnostic, not a proof of posterior convergence. ESS, retained draws, direction-grid approximation, and the fixed nominal error model remain limitations. The latest adjacent-size comparison above should be used to decide whether more samples are needed.
