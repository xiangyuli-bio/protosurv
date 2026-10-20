# Data schema

## Cohort CSV

`patient_id,slide_id,survival_time,event,feature_path` identifies one WSI per row.
IDs are read as strings, slide IDs are unique, time is positive/finite, and the
event indicator is exactly 0 or 1. Relative feature paths resolve against the
CSV directory. All slides for a patient must have identical outcome values.
Use one consistent clinical time unit and record release/linkage/exclusions.

## HDF5

- `features`: float32 `[N,D]`, frozen patch representation (default D=1024).
- `coords`: float32 `[N,2]`, patch-centre coordinates at level zero.
- `priors`: float32 `[N,5]`, ordered Tumor/Stroma/Immune/Necrosis/Others.
- `edge_index`: optional int64 `[2,E]`, no self-loops required in storage.
- `mpp` or `pixel_spacing` attribute: positive micrometres per level-zero pixel,
  scalar or `(x,y)` pair, required for physical radius matching.

Graph keys can be changed in the YAML. The sensitivity runner stores separate
fold/variant keys so one graph does not overwrite another graph's edge list.
The GCN adds self-loops. `prior_mode=hard` derives one-hot priors from the same
cached soft matrix at load time. The residual class remains in the model but
is excluded from the alignment target.

CONCH similarity embeddings and downstream visual features may have different
dimensions. The prior converter takes aligned image/text embeddings with the
same similarity-space dimension and stores `[N,5]` weights alongside the
configured visual features. Patch order must match exactly. Record image/text
projection choice, encoder checkpoint, transforms and prompts; no survival
labels enter this step. CONCH loading is provider-specific and upstream.

## Independent annotation NPZ

Use one reference file matching each exported slide filename:

- `tumor`: float `[N]` in `[0,1]`, tumor area fraction on the raster footprint.
- `patch_area`: positive float `[N]`, footprint area in common raster pixels.
- `tissue`: boolean `[N]`, complete independent reference coverage.
- `coords`: optional `[N,2]`, identical ordering to prediction coordinates.
- `patient_id`: optional scalar string for patient-clustered bootstrap.

The rasterizer consumes independently exported pathologist masks, never
CONCH pseudo-labels. It requires the true level-zero patch footprint size,
mask downsample and either a validity mask or an explicit fully-annotated
flag. It rejects overlapping footprints. Record official train/test inclusion
lists and mask export provenance. Only the Tumor family is scored.
