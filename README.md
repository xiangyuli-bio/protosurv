# Prior-guided WSI survival model

Review-stage reference implementation for soft histopathologic priors, tissue
prototypes, sparse graph context and patient-level Cox survival prediction.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,dev]'
pytest -q
ruff check protosurv scripts tests --select E9,F
```

The source contains no cohort data or restricted foundation-model weights.
Patch extraction and authorized CONCH image/text encoding are upstream steps.
The repository converts cached embeddings to priors and provides the core
training, inference and evaluation pipeline. Record the exact encoder release,
preprocessing, prompt set and feature manifests for every measured run.
`configs/prompts.json` matches the manuscript's three prompts per tissue class.

## Inputs

The metadata CSV has one row per WSI:

```text
patient_id,slide_id,survival_time,event,feature_path
```

All slides from a patient share one outcome and one split. Larger risk means
shorter expected survival. `event=1` denotes death; `event=0` denotes censoring.
HDF5 files contain `features [N,D]`, patch-centre `coords [N,2]`, five-class
`priors [N,5]`, and optional `edge_index [2,E]`. Classes are Tumor, Stroma,
Immune Infiltration, Necrosis, Others. See [data_schema.md](docs/data_schema.md).

## Fixed folds and three seeds

```bash
python -m scripts.make_splits --metadata /data/tcga_luad/metadata.csv \
  --config configs/tcga.yaml --output-dir /data/tcga_luad/splits
python -m scripts.train_cv --metadata /data/tcga_luad/metadata.csv \
  --config configs/tcga.yaml --splits-dir /data/tcga_luad/splits \
  --seeds 1 2 3 --output-dir results/tcga_luad_soft
```

Training writes `seed_1/fold_0_best.pt` etc., patient-level test predictions,
fold histories and a seed-averaged outer-fold mean/sample SD. Use
`configs/tcga_hard.yaml` for the hard-prior counterpart with exactly the same
patient folds. Splitting randomness is fixed independently of training seeds.
Checkpoint metadata records all source patient IDs and source-training risk
normalization statistics. The full training-fold Cox objective retains
activations; its training memory differs from single-slide inference memory.

## Revision evaluations

[revision_experiments.md](docs/revision_experiments.md) provides executable
commands for CPTAC transfer, paired patient bootstrap, k/radius retraining,
CAMELYON16 mask-based localization and RTX 3090 runtime profiling. The graph
builders use SciPy KD-trees. Radius matching requires physical pixel spacing
and uses training coordinates only.

Attention export writes raw prototype maps plus `class_attention [N,5]`.
For each checkpoint, prototypes are averaged within class and min-max
normalized before ensembling. The default localization evaluator consumes
`class_attention` without a second normalization. Constant maps have zero
scores. Legacy single-model `attention` inputs remain supported explicitly.

Independent annotations contain fractional tumor occupancy and patch area.
This preserves raster mask area instead of assigning one hard tumor label to
a mixed patch. Incomplete reference coverage is excluded. Calibration/test
partitions remain separate. Bootstrap intervals use patient IDs when supplied
and otherwise explicitly use slides as the resampling units.

## Scope and evidence

A synthetic smoke check establishes software execution, not external
prognostic performance. No illustrative clinical C-index, IoU or GPU latency
is supplied as a measured result. Published numeric results require actual
cohort inputs, fixed checkpoints, manifests and generated reports. GPU
operations can be nondeterministic across environments; save seeds, software
versions and artifact checksums with each run. See [NOTICE.md](NOTICE.md) and
[LICENSE](LICENSE) for source and third-party terms.
