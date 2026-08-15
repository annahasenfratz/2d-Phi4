# Lambda=0.022 fixed-eta 5x5 kernel optimization

## Convention
- Real-space 5x5 D4-symmetric kernel.
- `sum K = 1`.
- `eta = 0.25` fixed, so `2^(eta/2) = 1.0905077326652577`.
- Map: `psi = 2^(eta/2) K phi` on the full periodic L32 lattice.
- Blocked field: `psi[:,0::2,0::2]`; no four-sublattice average.

## Inputs
- Fine L32: `/Users/anna/Work/Research/Normalizing-flow/Inverse_RG/phi4_phase-diagram/ensembles/lam0p022_kappa0p271_L32_embedded_wolff_sign_cluster_plus_radial_heatbath/configs.npz`
- Direct L16: `/Users/anna/Work/Research/Normalizing-flow/Inverse_RG/phi4_phase-diagram/ensembles/lam0p022_kappa0p271_L16_embedded_wolff_sign_cluster_plus_radial_heatbath/configs.npz`
- compare_n: `1000`

## Selected Kernel
- best start: `random_5`
- D_op: `80.1299`
- local RMS z: `4.93809`
- IR RMS z: `3.73008`
- min |Ktilde| on L32 grid: `0.54845185`
- min |K_eta_tilde| on L32 grid: `0.59809099`

Shell weights:
- w00 = `0.84137606581`
- w10 = `-0.0460805781758`
- w11 = `-0.0145304939998`
- w20 = `0.0404317891657`
- w21 = `0.0355547739881`
- w22 = `-0.0112742814189`
- normalization check = `1`

## Files
- `kernel5x5_summary.json`
- `kernel_coefficients.json`
- `optimization_log.csv`
- `operator_matching_table.csv`
- `blocked_lam0p022_kappa0p2705_L32_to_L16_kernel5x5_fixedeta.npz`

The selected kernel is the fixed-eta real-space even-even kernel for the lambda=0.022 branch.
