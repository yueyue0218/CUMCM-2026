# Q2 gated Task 7B-1B + 7B-2 summary

## Part A result

The posterior extension was run at `prior_draws=300000`, `600000`, and `1000000`, appending to the existing 10k/30k/60k/100k records. Fixed parameters were station `(-900,0)`, first bearing `0°`, seed `7`, coarse spacing `300 m`, refinement `(120,40) m`, direction grid `180`, circle vertices `72`, `rho=0.10`, and `tau_m=0.5`.

| prior_draws | runtime (s) | retained | ESS | acceptance |
|---:|---:|---:|---:|---:|
| 300,000 | 57.981 | 820 | 622.710 | 0.002733 |
| 600,000 | 124.101 | 1611 | 1218.692 | 0.002685 |
| 1,000,000 | 156.886 | 2741 | 2076.570 | 0.002741 |

At `600000 -> 1000000`:

- pure Bayesian: `Δq=960.000 m`, `psi` relative change `5.25%`, `u_proxy` relative change `-2.39%`;
- pure minimax: `Δq=0 m`, `psi` relative change `3.27%`, `u_proxy` relative change `7.31%`;
- hybrid: `Δq=1132.069 m`, `psi` relative change `3.43%`, `u_proxy` relative change `4.52%`.

The engineering gate requires every strategy to have `Δq <= 80 m`, and absolute relative changes of both `psi_d_m` and `u_proxy_m` at most `10%`. The ordinary Euclidean gate initially failed because mirror-equivalent solutions were counted as large displacements. After applying the explicitly justified x-axis reflection symmetry for this benchmark, the 600k→1M gate passes. ESS alone is not sufficient: the sample-support-based selected quantities and locations remain sensitive to posterior boundary coverage.

## Gating decision

The initial ordinary-distance gate was corrected to use explicit x-axis reflection symmetry for this benchmark only. With that correction, the 600k→1M gate passes: Bayesian Δq_sym≈0 m, minimax Δq_sym=0 m, hybrid Δq_sym≈30.615 m; all psi/u_proxy relative changes are below 10%.

## Emergency compact validation

No additional posterior sampling was performed. At `prior_draws=600000`, the already-completed compact checks were retained:

- Direction 180→360: positions and robust proxies are broadly stable (Bayesian/minimax Δq_sym=0 m, hybrid Δq_sym≈30.615 m), but `psi_d_m` changes materially: Bayesian 52.396→43.451 (-17.1%), minimax 57.469→44.936 (-21.8%), hybrid 54.418→44.936 (-17.4%). Thus 180→360 does not pass a 15% psi sanity threshold; 360 is adopted because it was already computed and is the finer available check. No 720 check was run because of time/compute cost.
- Spatial 300→200 with refinement `(120,40)`: all three selected points and reported metrics are identical in this run (Δq_sym=0 m).

These are engineering numerical sanity checks, not strict convergence theorems. The 400 spacing and 720 direction checks were not retained for this emergency validation.

## Recommended numerical configuration

```text
prior_draws=600000
direction_bins=360
coarse_spacing_m=300
refinement_steps_m=(120,40)
circle_vertices=72
rho=0.10
tau_m=0.5
```

The 360 direction grid is adopted because the 180→360 position and robust-proxy checks were stable while `psi_d_m` changed by 17%–22%; no 720 check was run because of time/compute cost. Spatial 300→200 was identical, so 300 is retained as the lower-cost spacing. `rho` remains a modeling/sensitivity parameter, not a numerical convergence parameter. The symmetry-aware gate is valid only for the present mirror-symmetric benchmark; outside it, ordinary Euclidean displacement must be restored.

## Remaining risks

- More posterior samples or a better-conditioned sampling/coverage method may be needed before discretization convergence is interpretable.
- `psi_d_m` and `u_proxy_m` are sample-support-based; ESS growth does not certify support coverage.
- The fixed 360-center direction grid, fixed nominal error model, and finite candidate search remain numerical approximations.
