# Checkpoint and Restart Policy

Runs must checkpoint often enough to recover or extend without changing run
metadata.

## Checkpoint Contents

A restartable checkpoint should contain:

- current sweep;
- current chain state needed to continue;
- RNG state;
- acceptance counters;
- buffered observables if appending requires them;
- all parameters needed to continue consistently, or a pointer to
  `run_config.yaml`.

The canonical latest checkpoint path is:

```text
checkpoints/checkpoint_latest.pt
```

or:

```text
checkpoints/checkpoint_latest.npz
```

Sweep-specific checkpoints should use:

```text
checkpoints/checkpoint_sweep_<sweep>.pt
checkpoints/checkpoint_sweep_<sweep>.npz
```

## Extension Rules

The extension script must:

- take an existing run directory;
- read `run_config.yaml`;
- find `checkpoints/checkpoint_latest.*` or the newest
  `checkpoint_sweep_*.*`;
- append sweeps using the same run metadata;
- write `logs/extend_<timestamp>.log`;
- update `status.json`;
- avoid overwriting old observables without backup or append-safe logic.

## Raw Configs

Input/native/generated field configurations belong under
`data/configs_phi4_2d/`. Do not write raw generated ensembles under
`perfect_blocking_upsampling/`. Checkpoint state is the only exception, because
it is required to continue a run.

