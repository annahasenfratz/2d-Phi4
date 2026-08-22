# Scientific-results archive conventions

`registry/` is the project-wide, append-only index.  A `run_id` is stable and
points to a raw output directory; assigning it never authorizes renaming,
moving, or editing that directory.  Parameters in the registry must be read
from the run's `run_config.json`/`run_config.yaml` and completion state from
`status.json`, rather than inferred from a directory name.

`references.csv` identifies native ensembles and their observable summaries.
It separates the target reference from transformed or rethermalized samples.

Each question gets a directory under `studies/`.  Its `runs.csv` selects the
registry entries used for the result and records any valid combination rule.
Its `observables.csv` contains the derived, sweep-dependent measurements used
in the conclusion, with both value and uncertainty.  A study `README.md`
states the current scientific conclusion, scope, and known limitations.

When an important later analysis changes a conclusion, update that study's
`README.md` and append or revise the corresponding `observables.csv` rows.
Do not leave the only record in a terminal transcript or chat.  Preserve raw
outputs as immutable provenance; create new registry or study rows instead.
