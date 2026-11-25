# Inputs

`prepare_paired(rna, msi, xy, spot_ids=...)` accepts finite NumPy arrays with
matching rows: RNA (n × genes), MSI (n × peaks), coordinates (n × 2).
Spot identifiers must be unique. Pairing and registration are performed upstream.

For raw count/intensity matrices, use `normalize_counts`. Supply every measured
feature when calculating row totals, then select fixed features. Each row needs
a positive total. Do not normalize frozen figure inputs again.

Default graph construction uses spatial k=6, feature k=20, hyperedge sizes
(5, 20) including each center, and PCA dimension 30. Both coordinate axes need a
nonzero range. Correlation graphs require nonconstant rows; historical constant
rows are accepted only through an explicit compatibility option.

For precomputed operators, call `prepare` with `coordinates`,
`feature_graph_rna`, `feature_graph_msi`, `spatial_graph`, `causal_mask` and
`propagation`. Each graph is n × n. NumPy, torch and SciPy sparse graphs are
accepted; `prepare` validates and converts them without changing preprocessing.
