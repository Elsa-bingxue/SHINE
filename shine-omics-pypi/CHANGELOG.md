# Changelog

## 0.1.1 — 2026-11-27

- Separate cross-modal `local_k` from the spatial encoder graph and add an optional all-spot mask.
- Retain backward-compatible graph construction when the new parameters are omitted.
- Record the grouped-feature/dense-cross execution used in the matched mask experiment.
- Add four-neighbor striatal presets; retain six neighbors in the other sections.


## 0.1.0 — 2026-11-24

- Add the `SHINE` estimator: fit, transform, cluster, save and load.
- Add count normalization and unified construction of spatial, feature and hypergraph operators.
- Expose reconstruction-only loss, pre-fusion L2 and the spatial attention floor as parameters.
- Add section replay, evaluation and plotting helpers for seven figure sections.
- Preserve the earlier training modules and separate historical code archive.
- Include input, output and reproducibility guides; declare PyTorch as a dependency.

Saved historical weights replay exactly in the reference environment. Fresh
120-epoch validation reproduced six of seven displayed partitions. In mPD4,
an embedding difference of about 2e-6 changed KMeans assignments; the notebook
reports both results and retains the historical seed 123 display.

## 0.0.1

Initial package containing the original SHINE workflow.
