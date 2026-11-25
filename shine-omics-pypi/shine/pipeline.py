"""Explicit preprocessing and graph construction for paired spatial inputs.

Registration is upstream: this module never guesses correspondence or filters
spots using region labels. Reproduction uses frozen feature and spot orders.
"""
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from .api import prepare


def normalize_counts(matrix, *, target_sum=10000., feature_indices=None):
    """Total-count/TIC normalization and log1p, before selecting features.

    Supply the complete measured gene/peak matrix for the denominator. Zero
    totals raise instead of silently deleting spots from the paired cohort.
    Already normalized inputs must not be passed to this function again.
    """
    x = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    x = np.array(x, dtype=np.float32, copy=True)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Counts must be a finite nonnegative matrix.')
    totals = x.sum(1)
    if (totals <= 0).any() or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError('Positive row totals and target_sum are required.')
    if feature_indices is not None:
        x = x[:, feature_indices]
    return np.log1p(x / totals[:, None] * target_sum).astype(np.float32)


def _normalized_graph(adjacency):
    # Historical transpose convention is intentional, including on directed
    # feature kNN graphs. Spatial support already has self loops; adding I here
    # gives a weight-2 diagonal, exactly as in the final experiments.
    a = sp.coo_matrix(adjacency) + sp.eye(adjacency.shape[0])
    d = sp.diags(np.asarray(a.sum(1)).ravel() ** -.5)
    a = a.dot(d).transpose().dot(d).tocoo()
    return torch.sparse_coo_tensor(np.vstack([a.row, a.col]),
                                   a.data.astype('float32'), a.shape).coalesce()


def build_graphs(rna, msi, xy, *, spatial_k=6, feature_k=20,
                 hyperedge_sizes=(5, 20), pca_components=30, pca_seed=0,
                 sparse_propagation=False, allow_constant_feature_rows=False):
    """Build the revision's fixed H5+H20 and local self+6 operators.

    Hyperedge sizes include the center. PCA is fitted separately per modality,
    then each score matrix is divided by its Frobenius norm. No labels enter.
    Returns model arguments plus the incidence matrix and a compact audit.
    """
    rna, msi, xy = [np.asarray(v, dtype=np.float32) for v in (rna, msi, xy)]
    n = len(rna)
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in (rna, msi, xy)):
        raise ValueError('Provide finite two-dimensional matrices.')
    if len(msi) != n or xy.shape != (n, 2):
        raise ValueError('RNA, MSI and coordinates must have the same spot order.')
    if not 0 < min(spatial_k, feature_k) <= max(spatial_k, feature_k) < n:
        raise ValueError('Neighbor counts must be positive and smaller than n.')
    if not hyperedge_sizes or any(k <= 0 or k > n for k in hyperedge_sizes):
        raise ValueError('Invalid hyperedge sizes.')
    if not 0 < pca_components <= min(n, rna.shape[1], msi.shape[1]):
        raise ValueError('PCA dimension exceeds available observations/features.')
    if (np.ptp(xy, axis=0) == 0).any():
        raise ValueError('Both spatial coordinates need a nonzero range.')
    constant_counts = [int((np.std(x, axis=1) == 0).sum()) for x in (rna, msi)]
    if any(constant_counts) and not allow_constant_feature_rows:
        raise ValueError('Constant feature rows make correlation distance undefined.')
    coords = torch.tensor((xy - xy.min(0)) / np.ptp(xy, axis=0))
    mask = kneighbors_graph(xy, spatial_k, include_self=False).toarray().astype('float32')
    space = np.maximum(mask, mask.T).astype('float64')
    np.fill_diagonal(space, 1)
    np.fill_diagonal(mask, 1)
    parts = []
    for x in (rna, msi):
        z = PCA(pca_components, random_state=pca_seed).fit_transform(x)
        z /= max(np.linalg.norm(z), 1e-8)
        parts.append(z)
    joint = np.concatenate(parts, axis=1)
    incidence = sp.hstack([kneighbors_graph(joint, k, include_self=True).T
                           for k in hyperedge_sizes], format='csr').astype('float32')
    h = torch.tensor(incidence.toarray())
    v, e = (h.sum(1) + 1e-6).rsqrt(), (h.sum(0) + 1e-6).reciprocal()
    propagation = v[:, None] * ((h * e[None, :]) @ h.T) * v[None, :]
    if sparse_propagation:
        propagation = propagation.to_sparse().coalesce()
    return dict(coordinates=coords,
                feature_graph_rna=_normalized_graph(kneighbors_graph(rna, feature_k, metric='correlation')),
                feature_graph_msi=_normalized_graph(kneighbors_graph(msi, feature_k, metric='correlation')),
                spatial_graph=_normalized_graph(space), causal_mask=torch.tensor(mask),
                propagation=propagation, incidence=incidence,
                audit=dict(n_spots=n, spatial_k=spatial_k, feature_k=feature_k,
                           hyperedge_sizes=list(hyperedge_sizes), n_hyperedges=incidence.shape[1],
                           pca_components=pca_components, pca_seed=pca_seed,
                           constant_feature_rows=constant_counts,
                           allow_constant_feature_rows=allow_constant_feature_rows,
                           labels_used=False, sparse_propagation=sparse_propagation))


def prepare_paired(rna, msi, xy, *, spot_ids=None, **graph_options):
    """Build graphs from already paired/preprocessed matrices, then validate."""
    graphs = build_graphs(rna, msi, xy, **graph_options)
    model_inputs = {k: v for k, v in graphs.items() if k not in ('incidence', 'audit')}
    return prepare(rna, msi, spot_ids=spot_ids, **model_inputs), graphs
