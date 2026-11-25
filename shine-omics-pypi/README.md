# SHINE (`shine-omics`)

SHINE integrates paired spatial transcriptomics and mass spectrometry imaging
(MSI). It combines spatial and feature graphs, within-modality attention,
local cross-modal attention and hypergraph-derived propagation to learn a
shared spot representation.

## Install

```bash
python -m pip install --upgrade shine-omics
```

The distribution is `shine-omics`; import it as `shine`. Version 0.1.0 supports
Python 3.8–3.11 and installs PyTorch. PyTorch Geometric is not required by the
current API. For plotting and section notebooks:

```bash
python -m pip install 'shine-omics[analysis]'
```

CPU is the reference backend. CUDA requires a compatible PyTorch installation;
MPS is not supported. Numerical dependencies are pinned for reproducibility.

## Inputs

Provide three arrays with matching spot order:

- RNA: spots × genes, normalized and log-transformed;
- MSI: spots × peaks, normalized and log-transformed;
- coordinates: spots × 2, from the paired tissue section.

Registration, tissue QC and feature selection precede SHINE. For raw counts,
`normalize_counts(full_counts, feature_indices=...)` normalizes each spot to
10,000 and applies log1p, using all measured features for the denominator.
Do not apply it to matrices that are already normalized.

## Quick start

```python
from shine import SHINE, prepare_paired

# rna, msi and xy are paired, preprocessed arrays.
data, graphs = prepare_paired(rna, msi, xy, spot_ids=spot_ids)
model = SHINE(loss="original", l2=False, spatial_floor=0.0)
embedding = model.fit_transform(data, epochs=120, seed=123)
labels = model.cluster(n_clusters=11, seed=0)
model.save("shine.pt")
```

Default graph construction uses H5 + H20, spatial kNN6, feature
correlation-kNN20 and local self+6 cross-modal attention. PCA uses 30 components
per modality; each input must therefore have at least 30 features and spots.
Graph sizes and PCA dimension can be passed to `prepare_paired` for smaller data.
No annotation labels are used for training or graph construction.

## Model options

| Parameter | Default | Meaning |
|---|---|---|
| `loss` | `"original"` | Reconstruction, correspondence and contrastive losses. Use `"reconstruction"` for reconstruction alone. |
| `l2` | `False` | Normalize modality vectors before fusion. |
| `spatial_floor` | `0.0` | Minimum spatial-branch weight within each modality, between 0 and 1. |
| `latent_dim` | `128` | Final embedding dimension. |

All options retain hypergraph propagation and final ReLU. A spatial floor of
0.5 means at least 50% spatial weight within each modality, not equal RNA/MSI
weights. Training defaults to 120 epochs, learning rate 1e-4 and weight decay
1e-3; the final epoch is retained. Settings and seeds are chosen explicitly.

## Outputs

- `model.embedding_`: spot embeddings before clustering normalization;
- `model.labels_`: KMeans labels, after calling `cluster`;
- `model.attention_`: learned and effective within-modality weights;
- `model.history_`: training losses for every epoch.

Attention weights describe model mixing, not causal biological importance.
KMeans uses row-normalized embeddings and a separate seed; this normalization
is independent of pre-fusion `l2`.

```python
model = SHINE.load("shine.pt")
embedding = model.transform(data)
```

Checkpoints store parameters, seeds, weights and an input/graph fingerprint.
Replay requires the same input values, spot order and graphs. New sections need
a new fit. Load only trusted checkpoints; optimizer resume is not supported.

## Reproducibility

Seven section notebooks cover the SHINE results in Figures 2–5. Spatial panels
use training seed 123. Summaries use five training seeds (0, 1, 2, 123, 1234),
with repeated clustering averaged within each fit before computing mean and SD.
Section settings include optional L2, loss and spatial-weight parameters.

The source distribution includes `docs/`, `configs/` and a runnable synthetic
example. These are also installed under `share/shine-omics` in the Python
installation. Full datasets, trained checkpoints and section notebooks are
kept in the analysis project, outside the package.

## Scope and compatibility

The propagation operator is derived from hypergraph incidence:
`A_H = D_v^(-1/2) H D_e^(-1) H^T D_v^(-1/2)`.
It encodes multi-node neighborhoods but is a fixed linear operator before the
final ReLU; it does not imply greater expressivity than a GCN using the same
operator.

The earlier `shine.model`, `shine.preprocess` and `shine.Training` modules remain
available. The 120-epoch `original` objective above is distinct from the older
300-epoch dynamic-loss workflow. See the bundled advanced and reproducibility
guides for migration details and numerical limitations.

## Links

- [Source repository and data references](https://github.com/Elsa-bingxue/SHINE)
- [Issue tracker](https://github.com/Elsa-bingxue/SHINE/issues)
- [PyPI releases](https://pypi.org/project/shine-omics/)
