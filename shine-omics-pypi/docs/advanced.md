# Advanced and legacy APIs

Install `shine-omics>=0.1.0`. The simple `SHINE` API covers the adopted
figure configurations; the lower-level API exposes experimental training modes.

## Fusion normalization

The default is `normalize_fusion=False`. To give the RNA and attended MSI
vectors unit L2 norm at each spot before concatenation, pass
`normalize_fusion=True` to `shine.model.Encoder_overall`, or run
`python -m shine.Training --normalize-fusion` with the prepared input files.
Both configurations apply ReLU after hypergraph propagation.

## Optional revision training modes

The revision API is available as `shine.revision.RevisionSHINE`. It uses
the backbone and fixed objectives evaluated in the 120-epoch H5+20, local self+6
experiments. It does not change the older `shine.model` / `shine.Training`
300-epoch workflow or migrate existing figure runs automatically.

```python
from shine.revision import RevisionSHINE

model = RevisionSHINE(
    rna_dim=rna.shape[1], msi_dim=msi.shape[1], latent=128,
    training_mode="original",       # default
    normalize_fusion=False,         # independent optional pre-fusion L2
    spatial_floor=0.0,              # 0 keeps original within-modality attention
    correspondence_weight=1.0,      # 0 disables both correspondence directions
    infonce_weight=0.05,            # 0 disables InfoNCE independently
    retention_weight=0.1,           # used only by retention
)
out = model(
    rna, msi, coordinates, feature_graph_rna, feature_graph_msi, spatial_graph,
    causal_mask=local_self_plus_6_mask, propagation=H5_H20_propagation,
)
terms = model.loss_terms(out, rna, msi, spatial_graph)
terms["total"].backward()
# Construct the optimizer after the model; train all model.parameters().
# Save model.get_config() together with model.state_dict() and input/graph metadata.
```

| `training_mode` | Objective and gradient flow |
| --- | --- |
| `original` | Joint optimization with configurable alignment losses. Defaults: 2× RNA MSE + 2× MSI MSE, two correspondence terms, 0.05× symmetric InfoNCE (temperature 0.07). |
| `reconstruction_only` | Joint optimization: 2× RNA MSE + 2× MSI MSE. |
| `retention` | Reconstruction plus 0.1× each standardized frozen RNA/MSI teacher-feature MSE; two training-only linear heads. |
| `dgl` | Unimodal reconstruction trains each encoder; detached modality embeddings enter fusion, whose parameters receive fused reconstruction gradients. Two training-only GCN heads. This is the tested unsupervised DGL-style adaptation. |

### Optional within-modality spatial weight floor

`spatial_floor` is a finite number in `[0, 1]`, defaulting to `0.0`. For each
modality independently, it converts the learned spatial/feature weights to
`spatial_floor + (1 - spatial_floor) * spatial_weight` and
`(1 - spatial_floor) * feature_weight`. The two effective weights still sum to
one. `0.5` means **at least 50% spatial weight**, not a fixed 50:50 mixture or
an RNA/MSI mixing ratio. `1.0` uses only the spatial branch at this mixing step.

This option adds no parameters or prediction heads. Cross-modal attention,
hypergraph propagation, final ReLU and the separately selected losses/L2 remain.
`out["intra_rna"]` and `out["intra_msi"]` contain effective weights, in
`[spatial, feature]` column order; the corresponding `*_raw` entries retain the
unconstrained learned weights. Weights are not causal feature importance.

For the adopted LC091 Figure 4 configuration:

```python
model = RevisionSHINE(
    rna_dim=rna.shape[1], msi_dim=msi.shape[1],
    training_mode="reconstruction_only",
    normalize_fusion=False,
    spatial_floor=0.5,
)
```

The November 24 figure decision enables this only for LC091. BC_515_S1,
mPD3, mPD4 and the striatal sections retain `spatial_floor=0.0` and their
existing loss/L2 settings. This is an explicit dataset configuration; the API
does not select it automatically. Five-fit testing found improved LC091 whole
partition scores, worse breast scores, and a local/global tradeoff in nigra.
The seven adopted figure settings are recorded in
[`configs/revision_figures.json`](../configs/revision_figures.json). Pass a chosen
case's `model_kwargs` to `RevisionSHINE` along with the actual input dimensions;
clustering K is a separate evaluation parameter. Existing figures retain their
frozen training sources and embeddings.

Keep the floor fixed during training and inference, and save `get_config()`
with the checkpoint. It records `spatial_floor` and numerical execution options
(schema 4; the simple loader also reads schema 3 checkpoints); when loading a
pre-option configuration, use `config.get("spatial_floor", 0.0)`. Tensor state
keys and parameter counts are unchanged, so the tensor state alone cannot
identify the floor. Applying a different floor to an old trained model is not
equivalent to the reported retrained experiment. Frozen experimental files and
older 300-epoch entry points are retained; historical results remain tied to
their saved implementation and data.

### Independent alignment-loss parameters

For the BC_515_S1 and LC091 reconstruction-only configuration, use the same
backbone and set both alignment weights to zero:

```python
model = RevisionSHINE(
    rna_dim=rna.shape[1], msi_dim=msi.shape[1], latent=128,
    training_mode="original",
    normalize_fusion=False,
    correspondence_weight=0.0,
    infonce_weight=0.0,
)
```

This gives `2 * RNA_MSE + 2 * MSI_MSE`, with no auxiliary heads, teacher or DGL
gradient detachment. Attention, hypergraph propagation and final ReLU remain.
It is equivalent to the existing `training_mode="reconstruction_only"` mode.
By default, disabled losses are not evaluated and have no entries in the returned
loss dictionary. For exact historical LC091 reproduction, the frozen preset uses
`evaluate_zero_alignment=True` to retain the zero-weight gradient paths, together
with `attention_backend="grouped"`. These numerical compatibility settings do not
add a nonzero alignment penalty. `get_config()` records the resolved weights, including zeros, so
the mode name alone must not be used to identify the objective in result tables.

| Configuration in `original` mode | `correspondence_weight` | `infonce_weight` |
| --- | ---: | ---: |
| Revision baseline | 1.0 | 0.05 |
| Without correspondence | 0.0 | 0.05 |
| Without InfoNCE | 1.0 | 0.0 |
| Reconstruction only | 0.0 | 0.0 |

Weights must be finite and nonnegative. Omitted weights default to 1.0/0.05 in
`original` mode and 0/0 in all other modes. Nonzero alignment weights in the
other modes are rejected rather than silently changing their tested objectives.
The two correspondence directions currently share one coefficient. This does
not separately ablate RNA versus MSI correspondence.

The API does not infer tissue type or select weights from ARI/NMI. The cancer
configuration is an explicit experimental choice, not a claim that removing
alignment losses improves all cancer datasets or every endpoint. Existing runs
are unchanged; the regression suite verifies the zero-weight configuration
against the reconstruction-only implementation through optimizer updates.
Historical single-loss ablation runs evaluated inactive branches too, so their
dropout RNG schedule can differ from this API's skipped branches; their saved
results should retain their original experiment implementation as provenance.

For retention, first obtain final evaluation-mode RNA/MSI latents from a
separately trained, same-input/same-row/same-seed DGL teacher. Freeze and
standardize them once with `standardize_teacher_targets(teacher_rna, teacher_msi)`;
pass the returned target tuple as `teacher_targets=targets` to `loss_terms`.
Save the teacher checkpoint/spot-order provenance and returned standardization
statistics. Missing or trainable targets raise an error. A teacher fit is
additional computation, not part of the student's 120 epochs.

All modes retain the final ReLU. Original within-modality attention and local
cross-modal attention remain enabled. Mode and L2 are explicit training choices,
not automatic tissue-specific model selection; new combinations are untested
until evaluated. Inference uses `out["combined"]` (row-normalized for the existing
KMeans protocol); auxiliary heads and teachers are not required for inference.
Cross-attention weights use sparse edge order, not a dense spot-by-spot matrix.

Keep the same configuration across all independent training seeds for a dataset.
If configurations are selected using benchmark annotations, report that as
exploratory tuning rather than held-out performance. The default is not changed
to the winner of a particular endpoint.

Regression tests compare outputs, gradients and three optimizer steps against
the frozen experimental implementation, check DGL routing, frozen targets,
L2-on forwards/backwards, independent loss weights, zero-weight equivalence and
checkpoint round trips. They also cover the spatial-floor bounds, disabled
behavior, unchanged initialization, gradient flow and nonzero-floor checkpoint
round trips. Run from this directory:

```sh
SHINE_REFERENCE_EXPERIMENT=/path/to/revision_experiments/brain_dgl_noL2_20261118 \
  python -m unittest discover -s tests -p 'test_revision_modes.py' -v
```

## Notes
PyTorch is installed with the package. The current API does not require PyTorch Geometric.
