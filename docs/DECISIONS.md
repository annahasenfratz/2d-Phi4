## 2026-08-21 — Rethermalization algorithm for cascade

**Decision:** HMC rethermalization is abandoned for cascade production and replaced by Wolff+radial updates.

**Reason:** THERM-L64-LAM1 shows that HMC equilibrates local observables but does not reliably equilibrate long-distance observables over 500 sweeps, while Wolff+radial does.

**Affects:** phi4 inverse-blocking cascade production runs.

**Evidence:** [studies/thermalization/THERM-L64-LAM1/](../studies/thermalization/THERM-L64-LAM1/)
