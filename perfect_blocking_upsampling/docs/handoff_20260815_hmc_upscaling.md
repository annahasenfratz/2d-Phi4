# Handoff — 15 Aug 2026: HMC upscaling and large-volume checks

## Reboot status

There are **no incomplete production runs requiring continuation**.  The following recently watched runs are complete:

| stage/run | status | final HMC acceptance |
|---|---|---:|
| `L32toL64_N1500_S100_start0_HMCd4_n20_eps0p08_r24` | complete | 0.91454 |
| `L64toL128_N1500_S100_start0_HMCd8_n20_eps0p08_r25` | complete | 0.91458 |
| `L8toL512_N25_start0_HMC100_d4_eps0p025_tau2_b2_r1` | complete | 0.93913 |

After reboot, do not try to resume these runs.  Their final fields are already written as `final_phi.npz` in the run directories.

## Current active method

The flow is used **once**, to initialize the finer field.  All subsequent rethermalization is direct fine-field HMC targeting `exp(-S_f(phi))`:

- no latent `z` refresh after sweep zero;
- no coarse action or inverse kernel after sweep zero;
- HMC has independent Gaussian momenta on the active residue class;
- second-order leapfrog and a full fine-action plus kinetic-energy accept/reject step;
- a full sweep updates every residue class once, so every fine site is touched once.

Active flow/kernel pair:

```text
flow: perfect_blocking_upsampling/runs/lam1p0/training/
      lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/
      checkpoints/checkpoint_best_nll.pt

kernel: perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/
        alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json
```

### Successful L16 -> L32 HMC test

`L16toL32_N1500_S100_start0_HMCd2_n20_eps0p08_r23` is the reference setup:

- 1500 chains, `start0`, 100 sweeps;
- `divide=2`, hence 4 residue classes and 256 active sites per trajectory;
- `step_size=.08`, `leapfrog_steps=20`, so trajectory length `1.6`;
- input: the thermalized L16 field from the earlier L8->L64 chain;
- output path:
  `perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/L16toL32/L16toL32_N1500_S100_start0_HMCd2_n20_eps0p08_r23/`.

It thermalized more smoothly than the older `.08 x 10` (`tau=.8`) stage.  Keep the same `.08 x 20` parameters for the current larger-stage tests.

## New larger-stage runs

Both use the same HMC integrator `.08 x 20`, trajectory length 1.6, and retain a fixed `16^2=256` active sublattice per trajectory:

| stage | run | divide | residue classes | final field |
|---|---|---:|---:|---|
| L32->L64 | `L32toL64_N1500_S100_start0_HMCd4_n20_eps0p08_r24` | 4 | 16 | `.../L32toL64/...r24/final_phi.npz` |
| L64->L128 | `L64toL128_N1500_S100_start0_HMCd8_n20_eps0p08_r25` | 8 | 64 | `.../L64toL128/...r25/final_phi.npz` |

Their standard notebook locations are, respectively:

```text
.../L32toL64/L32toL64_N1500_S100_start0_HMCd4_n20_eps0p08_r24/levels/L32toL64/
.../L64toL128/L64toL128_N1500_S100_start0_HMCd8_n20_eps0p08_r25/levels/L64toL128/
```

The generic multi-stage runner creates `levels/L...toL...`.  The standalone fine-HMC runner also exposes a `levels/L16toL32` symlink.  Analysis notebooks should always point to `CHAIN_DIR / "levels" / LEVEL_NAME`.

Naming convention: retain `L{coarse}toL{fine}_N..._S..._start..._HMCd..._n..._eps..._r#`; do not add descriptive words such as `L8toL16` to the run name when those fields are only the input.

## Recent analysis results

### Critical susceptibility scaling

For direct native L=8,16,32,64,128, using 20-configuration bins:

```text
chi = G(0) = L^2 [<m^2>-<m>^2]  ~ L^(2-eta)
fit L=32,64,128: eta = 0.254(6)
```

This agrees very well with the 2D Ising value `eta=1/4`.

### Transverse-zero-momentum spatial correlator

The QCD-style projected correlator was formed from `P(x)=sum_y phi(x,y)`:

```text
C_py0(x) = (1/L) sum_x0 <P(x0+x) P(x0)>.
```

The fit accounts for periodic wrap-around:

```text
C_py0(x) = A [ exp(-m x) + exp(-m (L-x)) ].
```

Using the conservative long-distance range `L/4 <= x <= L/2` gives:

| L | m | m L |
|---:|---:|---:|
| 32 | 0.03118(62) | 0.998(20) |
| 64 | 0.01593(42) | 1.020(27) |
| 128 | 0.00805(21) | 1.030(27) |

Plots and generator:

```text
perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/analysis/
  py0_correlator_direct_L32_L64_L128.pdf
perfect_blocking_upsampling/scripts/plot_lam1p0_py0_correlator.py
```

### Second-moment correlation length

The correct estimator is

```text
xi_2nd/L = sqrt(G(0)/G(p_min)-1) / [2 L sin(pi/L)].
```

Direct native values (20-config bin jackknife):

| L | xi_2nd/L |
|---:|---:|
| 16 | 0.883(11) |
| 32 | 0.892(8) |
| 64 | 0.881(12) |
| 128 | 0.888(12) |

Latest flowed + HMC sweep-100 values:

| L | xi_2nd/L |
|---:|---:|
| 32 (`r23`) | 0.883(19) |
| 64 (`r24`) | 0.916(20) |
| 128 (`r25`) | 0.907(20) |

The universal 2D-Ising value supplied by the user is `0.9050488292`.  The rethermalized values are compatible with both direct native and this value at their current errors.

### Cascade plot correction

`perfect_blocking_upsampling/scripts/plot_lam1p0_hmc_cascade.py` was corrected on 15 Aug:

- it previously had an **extra factor of L** in the `xi_over_L` denominator;
- it now uses `sqrt(G0/Gp-1)/(2*L*sin(pi/L))`;
- the xi panel now includes a dotted black line at `0.9050488292`.

Regenerated figure:

```text
perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/analysis/
  cascade_sweeps_k0340301.pdf
```

## Relevant submit scripts

```bash
# reference L16 -> L32 .08 x 20 HMC
bash perfect_blocking_upsampling/scripts/submit_lam1p0_L16toL32_hmc_n20_eps0p08.sh rNUMBER --execute --background

# L32 -> L64 .08 x 20 HMC; r24 has completed
bash perfect_blocking_upsampling/scripts/submit_lam1p0_L32toL64_hmc_n20_eps0p08.sh rNUMBER --execute --background

# L64 -> L128 .08 x 20 HMC; r25 has completed
bash perfect_blocking_upsampling/scripts/submit_lam1p0_L64toL128_hmc_n20_eps0p08.sh rNUMBER --execute --background
```

The L64->L128 launcher defaults to consuming the L32->L64 `r24` final field.  If using a different L32->L64 run, update the config `initial_source` or make a matching configuration before launching.
