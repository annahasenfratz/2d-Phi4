# Critical Reference Note

The bundled Ising references used for the perfect-blocking test are:

- `external/mlneuralsampler_multilevel/data/config/Ising_data_nx8_beta0.4400000000_data1000000.dat`
- `external/mlneuralsampler_multilevel/data/config/Ising_data_nx16_beta0.4400000000_data1000000.dat`

The corresponding zip archives are:

- `external/mlneuralsampler_multilevel/data/config/ising8x8.zip`
- `external/mlneuralsampler_multilevel/data/config/ising16x16.zip`

The project now also generates critical Wolff-cluster references at
`beta_c = 0.5 * log(1 + sqrt(2))` with `500` configurations per volume and
saves them to:

- `perfect_blocking_ising/outputs/critical_ising_L8.npy`
- `perfect_blocking_ising/outputs/critical_ising_L16.npy`
- `perfect_blocking_ising/outputs/critical_ising_L8_validation.npy`

For the earlier demo sanity check, an independent `8x8` sample at `beta=0.44`
was also written to:

- `perfect_blocking_ising/outputs/ising_beta044_generated_L8.npy`

The main report and summary are written under `perfect_blocking_ising/outputs/`.
