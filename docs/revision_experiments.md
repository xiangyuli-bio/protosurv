# Revision experiments: commands and required outputs

Experimental claims require generated reports from real inputs.
The source package's synthetic checks do not supply clinical or RTX 3090 data.
All commands run from the repository root after installation.

## 1. Fixed source models and independent CPTAC-LUAD transfer

Train the five TCGA-LUAD folds with three seeds as in the README. Run the hard
counterpart using `configs/tcga_hard.yaml` on the same fixed patient folds.

```bash
python -m scripts.evaluate_external \
  --metadata /data/cptac_luad/metadata.csv --config configs/tcga.yaml \
  --checkpoints results/tcga_luad_soft/seed_*/fold_*_best.pt \
  --bootstrap 1000 --output results/cptac_luad_soft.json
python -m scripts.evaluate_external \
  --metadata /data/cptac_luad/metadata.csv --config configs/tcga_hard.yaml \
  --checkpoints results/tcga_luad_hard/seed_*/fold_*_best.pt \
  --bootstrap 1000 --output results/cptac_luad_hard.json
python -m scripts.compare_external_predictions \
  --reference results/cptac_luad_hard.patients.csv \
  --candidate results/cptac_luad_soft.patients.csv \
  --output results/cptac_soft_minus_hard.json
```

The JSON reports contain patient/slide/event counts, observed-time range,
reverse-KM median follow-up, C-index and 95% patient-bootstrap CI. If the
reverse-KM median is unreached, report that explicitly. The paired comparison
uses identical patient bootstrap indices for both methods. Source-patient
IDs and source risk-normalization statistics are mandatory in checkpoints.
Record the exact clinical release, identifier linkage, time units, eligible
patient list and exclusions. Run baseline external predictions on this same
patient list with identical ensembling; baseline model implementations are
obtained from the respective published source packages and are not silently
substituted by the proposed model.

## 2. k/radius sensitivity on all five cohorts

```bash
python -m scripts.run_graph_sensitivity \
  --metadata /data/tcga_luad/metadata.csv --config configs/tcga.yaml \
  --splits-dir /data/tcga_luad/splits --seeds 1 2 3 \
  --output-dir results/graph_sensitivity_luad
```

Repeat for KIRC, KIRP, STAD and UCEC. This retrains k=4,8,12,16 and radius
variants, retains separate graph keys and selects the radius on training
slides in each fold. HDF5 coordinates are level-zero centres; `mpp` or
`pixel_spacing` is required. The output contains fold-specific radii in
micrometres, non-self degree summaries, per-fold/seed metrics, mean and sample
SD over the five seed-averaged test-fold estimates. Patient splits are fixed
across all variants. Do not choose the best k using outer test outcomes.

## 3. CAMELYON16 localization with independent reference masks

Export independent official annotation masks at a recorded raster scale.
Do not use model-generated labels as the reference. Generate one annotation
NPZ per retained feature file, separately for official training and test
partitions. Example values below describe file geometry and must be replaced
with the actual level-zero footprint and mask scale:

```bash
python -m scripts.rasterize_annotations \
  --features /data/camelyon16/slide.h5 \
  --tumor-mask /data/camelyon16/reference/slide_tumor.png \
  --valid-mask /data/camelyon16/reference/slide_valid.png \
  --patch-size-level0 512 --mask-downsample 32 \
  --output /data/camelyon16/annotations/test/slide.npz
python -m scripts.export_attention \
  --metadata /data/camelyon16/test.csv --config configs/tcga.yaml \
  --checkpoints results/tcga_luad_soft/seed_*/fold_*_best.pt \
  --output-dir results/camelyon16_soft/test
python -m scripts.evaluate_localization \
  --predictions results/camelyon16_soft/test \
  --annotations /data/camelyon16/annotations/test \
  --calibration-predictions results/camelyon16_soft/train \
  --calibration-annotations /data/camelyon16/annotations/train \
  --num-prototypes 8 --score-key class_attention \
  --bootstrap 1000 --output results/camelyon16_soft.json
```

Export the calibration maps separately using the official training CSV.
Repeat with fixed hard-prior checkpoints and the hard configuration.
For the original CONCH prior use `--score-key priors`. Record the actual
calibration/test positive/negative counts, global method thresholds, raster
provenance and exclusions. `class_attention` is already normalized within
each checkpoint before ensembling and must not be normalized a second time.
Add `patient_id` to reference NPZ files when known so uncertainty is clustered
by patient; otherwise describe the bootstrap as slide-level. Fractional tumor
occupancy preserves mixed patch boundaries. The report supplies tumor IoU,
precision over exactly 20% of evaluated tissue area, negative-slide area and
bootstrap intervals. It does not validate causal survival attribution.

## 4. Downstream RTX 3090 runtime

Use a cached real slide with at least 40,000 tissue patches and its matching
preconstructed graph. Run on one RTX 3090 with the actual trained checkpoint:

```bash
python -m scripts.benchmark --config configs/tcga.yaml \
  --checkpoint results/tcga_luad_soft/seed_1/fold_0_best.pt \
  --input-h5 /data/runtime/slide_with_graph.h5 \
  --patch-counts 5000 10000 20000 40000 --warmup 20 --repeats 100 \
  --output results/rtx3090_runtime.json
```

Set the YAML device to the intended CUDA device. The command uses increasing
prefixes of the same feature file and induced edge subsets; record this
sampling rule and realized edge counts. It synchronizes CUDA around each run
and reports median/IQR latency, all individual timings and peak allocated
memory in decimal GB (1 GB = 10^9 bytes). Features, model and graph are resident before timing. Encoder
inference, graph construction and data transfer are excluded. Training peak
memory is a separate measurement. A run without `--input-h5` is labelled
synthetic; a run without `--checkpoint` is labelled randomly initialized.
Neither should be described as trained-model real-slide performance.

## 5. Reviewer-accessible code

Upload the prepared source archive to an anonymous repository. Verify access
in a signed-out browser and record an immutable release/commit and archive
SHA-256. Cite the verified repository address only after it is accessible to reviewers.
The local archive is ready for hosting but does not create a public link.
