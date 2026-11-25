# Outputs

| Attribute | Content |
|---|---|
| `embedding_` | n × latent_dim float array, before row normalization |
| `labels_` | n KMeans labels, available after `cluster` |
| `attention_` | `rna`, `msi`, `rna_raw`, `msi_raw`; n × 2 arrays, columns spatial/feature |
| `history_` | One loss record per training epoch |
| `training_config_` | Training settings and seed |
| `input_info_` | Input shape, spot order and graph fingerprint |

`fit` returns the estimator. `fit_transform` returns an embedding copy.
`transform` replays a fitted model on the same data and rejects mismatched
inputs. `save` stores inference state and results; `load` does not resume the
optimizer. Refitting starts a new model and clears earlier cluster labels.

The optional `shine.plotting` module provides spatial/UMAP, regional, molecular
and attention panels. `save_panel` exports PDF, SVG and PNG. Molecular maps show
measured values; model association alone does not identify an MSI peak.
