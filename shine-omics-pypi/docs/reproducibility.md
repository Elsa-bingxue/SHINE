# Reproducibility

## Figure settings

`configs/revision_figures.json` records the adopted seven-section settings.
Every fit uses 120 epochs and final ReLU. L2, loss, spatial attention floor and cross-modal support are independent options.
The updated striatal figure uses four other attention neighbors; the remaining
sections use six. The spatial encoder graph stays at six neighbors. The settings were explored on these datasets and are
not held-out model-selection results.

The section notebooks start from frozen, preprocessed paired matrices. They
verify source hashes and replay historical weights. Raw-image registration is
outside this validation. New training is opt-in and saved separately.

Training seeds: 0, 1, 2, 123, 1234. Average KMeans seeds 0–4 within each fit,
then calculate the mean and sample SD across five fits. Spatial panels use
training seed 123, with clustering seed 123 for nigra and 0 for other sections.
Region scores subset the original whole-section cluster labels; regions are
not reclustered.

## Numerical behavior

The historical environment uses Python 3.8.20, PyTorch 1.8.0 and CPU execution;
key numerical dependencies are listed in `reference_constraints.txt`.
`attention_backend` and `evaluate_zero_alignment` preserve historical numerical
execution and are recorded in section settings. They are not additional
biological model choices.

For the previous six-neighbor reference manifest, saved-model replay matched
embeddings and labels in all seven sections. Seven fresh seed 123 fits completed
120 epochs. Six matched historical partitions;
mPD4 differed by 2.0e-6 in its embedding but KMeans selected a different
partition (partition agreement ARI 0.756). Its original and fresh metrics are
reported separately. Neither numerical equality across environments nor
identical clustering after retraining is guaranteed.

The updated A1/B1/C1 four-neighbor checkpoints replay exactly in the tested
environment, including the normalized embeddings and displayed cluster IDs.
They use grouped feature attention with native dense cross-attention
(`attention_backend="grouped_feature"`). This check replays completed weights;
it is separate from a new end-to-end training run.

## Propagation

The hypergraph incidence matrix defines a fixed normalized propagation
operator. This uses multi-node neighborhood information but is algebraically
identical to graph propagation with the same operator. No unique nonlinear
higher-order expressivity is claimed for this implementation.

## Legacy code

The earlier modules remain importable. The old 300-epoch dynamic loss and the
120-epoch revision objective are different. Historical results retain their
original source snapshots; the local analysis project also includes an archive
of code preceding the notebook migration.
