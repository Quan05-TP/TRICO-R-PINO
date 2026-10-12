# Development rules

Work on the maintained package in `src/pure_pino`, with examples and meaningful
regression tests. Use Python 3.10+ and the declared development dependencies.

Before a commit, run `python -m ruff check .`,
`python -m ruff format --check .`, `python -m pytest`,
`python -m pure_pino doctor`, and the external-only example.

Preserve target information boundaries: target internal temperature/depth must
not enter inputs, initialization, normalization, model selection or fine-tuning.
Original lake roles, aliases, observation clocks and missing-data denominators
must remain explicit. Do not insert the old temperature integrator into the
direct primary forward.

Native and feature dependencies are immutable source snapshots excluded from
automatic formatting. If a law changes, review its source manifest, update the
bound hashes and tests, and require fresh real-data physical qualification.
Never infer scientific admission from a software test, fixture score or a
nonzero gradient. The current split-clock release remains untrained.

Keep credentials, private machine configuration, raw data, databases, model
weights and experiment logs outside Git. Data-specific training requires its
own audited manifest, explicit budget and frozen evaluation protocol.
