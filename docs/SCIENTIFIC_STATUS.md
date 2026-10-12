# Scientific status

This package contains the latest `LayerNorm_split_SI_clock` direct neural
operator candidate. The new clock representation has no trained checkpoint or
validated population accuracy. Previously trained LayerNorm checkpoints use a
different feature contract and must not be presented as weights for this
candidate.

The published modules are a package port of the source snapshot identified in
`SOURCE_PROVENANCE.json`. Internal imports and dependency-file metadata were
changed for portability. The published package does not inherit an earlier
complete scientific qualification; fresh qualification must bind its actual
code, configuration, native dependencies and external data. Software tests or
a manufactured example establish engineering behavior, not lake accuracy.

## Model and physical scope

External features condition deterministic initial latent states and learned
block carries. The network directly queries a temperature/state trajectory;
its main forward does not call the legacy 900-second water-temperature solver.
The native modules provide constitutive relations, radiation partitioning,
bed exchange and surface-root calculations. Power quadrature used to check
inventory residuals is distinct from integrating water temperature.

Spatial support is a single-column, piecewise-constant P0 representation, with
native storage volumes and separate interface areas. It is not an arbitrary
continuous three-dimensional temperature field. Learned memory is a latent
representation, not a separately validated physical heat-history inventory;
weather-history context is not a reconstructed temperature spin-up.

The phase and coupled components use the explicit conditional reference-water
and compensated fixed-reference-liquid-volume model. Unknown chemistry is not
measured freshwater or zero salinity. Complete saline-water physics, measured
river carriers, moving water level, ice draft/freeboard, strict inventory
conservation, phase-trajectory accuracy and certified thermocline localization
are not established by this code release. A legal state range or a converged
local skin root does not establish a physically accurate lake trajectory.

## Information boundary

Target-lake internal temperature and sampled depth must not enter model
features, initialization, geometry construction or model selection. Geometry
must have independent external provenance. Source TRAIN observations may enter
supervision; target observations may only enter a separately frozen downstream
readout/scoring protocol. The package does not include an observation-driven
target initializer, target-specific optimization or a target-label dataset.

Historical reconstruction may use later external records under its declared
information cutoff. It is distinct from strictly causal or operational
forecast use. Preserve raw SI support clocks, availability masks and the UTC
feature contract. A new clock-candidate configuration must explicitly select
`RELATIVE_SUPPORT_SPLIT_SI_AFFINE_CLOCK_V1` and
`split_native_support_affine`; legacy configuration compatibility does not
authorize silently changing a checkpoint's feature units.

## External assets and interpretation

Large meteorological, geometry and canonical observation datasets, checkpoints,
private server paths and execution receipts are not part of this source
example. Reproducing a real experiment requires separately acquired,
hash-bound external support, a frozen source/target protocol and fresh
qualification. Missing or unsupported records must remain in coverage and
failure ledgers; they must not be replaced by synthetic data or zero error.

This release is an editable scientific code example. It does not claim that
the complete inverse-problem plan, full 281-lake scientific acceptance,
calibrated uncertainty or an accuracy improvement has been achieved.
