"""Small training API over the validated revision backbone.

Inputs are already paired and preprocessed. Graphs remain explicit so wrapping
an existing experiment cannot silently change its cohort or graph construction.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .revision import RevisionSHINE

_INPUTS = ('rna', 'msi', 'coordinates', 'feature_graph_rna',
           'feature_graph_msi', 'spatial_graph', 'causal_mask', 'propagation')
_GRAPHS = _INPUTS[3:]


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer.')


def _seed(value):
    if type(value) is not int or not 0 <= value < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32).')


def _tensor(value):
    if sp.issparse(value):
        value = value.tocoo()
        result = torch.sparse_coo_tensor(
            np.stack((value.row, value.col)), value.data, value.shape)
    elif torch.is_tensor(value):
        if value.requires_grad:
            raise ValueError('Prepared inputs and graphs must be fixed tensors.')
        result = value.detach()
    else:
        result = torch.tensor(np.asarray(value))
    if result.is_complex():
        raise ValueError('Inputs must be real-valued.')
    if result.layout not in (torch.strided, torch.sparse_coo):
        result = result.to_sparse_coo()
    result = result.to(device='cpu', dtype=torch.float32).clone()
    if result.is_sparse:
        result = result.coalesce()
    values = result.values() if result.is_sparse else result
    if result.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError('Each input must be a finite two-dimensional matrix.')
    return result


class PreparedData:
    """Validated tensors in a single fixed spot order; construct with prepare()."""

    def __init__(self, tensors, spot_ids):
        self._tensors = tensors
        self.spot_ids = tuple(spot_ids) if spot_ids is not None else None

    @property
    def shape(self):
        return (len(self._tensors['rna']), self._tensors['rna'].shape[1],
                self._tensors['msi'].shape[1])

    def fingerprint(self):
        """Hash effective float32 inputs, all graphs, and any supplied spot IDs."""
        digest = hashlib.sha256()
        digest.update(json.dumps(self.spot_ids).encode('utf-8'))
        for name in _INPUTS:
            value = self._tensors[name]
            digest.update(json.dumps([name, list(value.shape), value.is_sparse]).encode())
            arrays = (value.indices(), value.values()) if value.is_sparse else (value,)
            for array in arrays:
                digest.update(array.contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _on(self, device):
        return {name: value.to(device) for name, value in self._tensors.items()}


def prepare(rna, msi, coordinates, *, feature_graph_rna, feature_graph_msi,
            spatial_graph, causal_mask, propagation, spot_ids=None):
    """Bundle already processed inputs without scaling, filtering or rebuilding.

    Matrices may be NumPy arrays, torch tensors or SciPy sparse graphs. RNA/MSI
    and coordinates must be dense; all graphs must be square in the same spot
    order. ``propagation`` is the already computed hypergraph-induced A_H.
    The caller remains responsible for matching feature and coordinate rows.
    """
    values = dict(rna=rna, msi=msi, coordinates=coordinates,
                  feature_graph_rna=feature_graph_rna,
                  feature_graph_msi=feature_graph_msi, spatial_graph=spatial_graph,
                  causal_mask=causal_mask, propagation=propagation)
    tensors = {name: _tensor(value) for name, value in values.items()}
    n = len(tensors['rna'])
    if n < 2 or any(tensors[k].is_sparse for k in _INPUTS[:3]):
        raise ValueError('Provide at least two spots and dense RNA/MSI/coordinates.')
    if (len(tensors['msi']) != n or tensors['coordinates'].shape != (n, 2)
            or min(tensors['rna'].shape[1], tensors['msi'].shape[1]) < 1):
        raise ValueError('RNA, MSI and [spots, 2] coordinates must have aligned rows.')
    for name in _GRAPHS:
        graph = tensors[name]
        vals = graph.values() if graph.is_sparse else graph
        if graph.shape != (n, n) or torch.any(vals < 0):
            raise ValueError(name + ' must be a nonnegative [spots, spots] matrix.')
        if graph.is_sparse:
            rows = graph.indices()[0, graph.values().ne(0)].unique()
        else:
            rows = graph.ne(0).any(dim=1).nonzero(as_tuple=False).flatten()
        if len(rows) != n:
            raise ValueError(name + ' has empty rows; use the experiment graph with self support.')
    if spot_ids is not None:
        spot_ids = tuple(str(value) for value in spot_ids)
        if len(spot_ids) != n or len(set(spot_ids)) != n:
            raise ValueError('spot_ids must be unique and match the input rows.')
    return PreparedData(tensors, spot_ids)


class SHINE:
    """Fit the revision model with a small, explicit set of options.

    ``loss='original'`` means the 120-epoch revision's fixed loss combination;
    ``loss='reconstruction'`` disables correspondence and InfoNCE. ``l2`` applies
    before fusion, independently of optimizer weight decay. Both retain ReLU.
    Advanced DGL/retention experiments remain in shine.revision.RevisionSHINE.
    """

    def __init__(self, *, loss='original', l2=False, spatial_floor=0.0,
                 latent_dim=128, correspondence_weight=None, infonce_weight=None,
                 attention_backend='sparse', evaluate_zero_alignment=False):
        if attention_backend not in ('sparse', 'dense', 'grouped', 'grouped_feature'):
            raise ValueError('Unknown attention execution backend.')
        if type(evaluate_zero_alignment) is not bool:
            raise TypeError('evaluate_zero_alignment must be bool.')
        if loss not in ('original', 'reconstruction'):
            raise ValueError('loss must be "original" or "reconstruction".')
        if type(l2) is not bool:
            raise TypeError('l2 must be a bool.')
        _positive_int('latent_dim', latent_dim)
        if latent_dim % 2:
            raise ValueError('latent_dim must be even.')
        for name, value in [('spatial_floor', spatial_floor),
                            ('correspondence_weight', correspondence_weight),
                            ('infonce_weight', infonce_weight)]:
            if value is None and name != 'spatial_floor':
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(name + ' must be finite and nonnegative.')
        if spatial_floor > 1:
            raise ValueError('spatial_floor must be at most 1.')
        if loss == 'reconstruction' and any(v not in (None, 0) for v in
                                             (correspondence_weight, infonce_weight)):
            raise ValueError('Reconstruction-only training requires zero alignment weights.')
        self._params = dict(loss=loss, l2=l2, spatial_floor=float(spatial_floor),
                            attention_backend=attention_backend,
                            evaluate_zero_alignment=evaluate_zero_alignment,
                            latent_dim=latent_dim, correspondence_weight=correspondence_weight,
                            infonce_weight=infonce_weight)

    def get_params(self):
        return dict(self._params)

    def _make_model(self, rna_dim, msi_dim):
        p = self._params
        return RevisionSHINE(
            rna_dim, msi_dim, latent=p['latent_dim'],
            training_mode='original' if p['loss'] == 'original' else 'reconstruction_only',
            normalize_fusion=p['l2'], spatial_floor=p['spatial_floor'],
            attention_backend=p['attention_backend'],
            evaluate_zero_alignment=p['evaluate_zero_alignment'],
            correspondence_weight=p['correspondence_weight'], infonce_weight=p['infonce_weight'])

    def fit(self, data, *, epochs=120, seed=123, lr=1e-4, weight_decay=1e-3,
            device='cpu', progress=None):
        """Start one fresh fit; keep the final epoch, without label-based selection.

        Returns self. This is not a resume operation. history_ stores pre-update
        training losses; embedding_ and attention_ come from final eval mode.
        """
        if not isinstance(data, PreparedData):
            raise TypeError('Call prepare() to construct the paired data first.')
        _positive_int('epochs', epochs)
        _seed(seed)
        for name, value in [('lr', lr), ('weight_decay', weight_decay)]:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (name == 'lr' and value == 0)):
                raise ValueError(name + ' is invalid.')
        device = torch.device(device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Use cpu or cuda; this sparse backend does not support MPS.')
        inputs = data._on(device)
        n, rna_dim, msi_dim = data.shape
        devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
        history = []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            model = self._make_model(rna_dim, msi_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(**inputs)
                terms = model.loss_terms(output, inputs['rna'], inputs['msi'], inputs['spatial_graph'])
                if not torch.isfinite(terms['total']):
                    raise FloatingPointError('Non-finite loss at epoch ' + str(epoch + 1))
                terms['total'].backward()
                optimizer.step()
                history.append(dict(epoch=epoch + 1, **{k: float(v.detach()) for k, v in terms.items()}))
                if progress is not None:
                    progress(dict(history[-1]))
            model.eval()
            with torch.no_grad():
                output = model(**inputs)
        embedding = output['combined'].detach().cpu().numpy().copy()
        if not np.isfinite(embedding).all():
            raise FloatingPointError('Final embedding contains non-finite values.')
        self.model_ = model
        self.embedding_ = embedding
        self.attention_ = {key: output[name].detach().cpu().numpy().copy()
                           for key, name in [('rna', 'intra_rna'), ('msi', 'intra_msi'),
                                             ('rna_raw', 'intra_rna_raw'), ('msi_raw', 'intra_msi_raw')]}
        self.history_ = history
        self.input_info_ = dict(shape=[n, rna_dim, msi_dim], fingerprint=data.fingerprint(),
                                spot_ids=list(data.spot_ids) if data.spot_ids is not None else None)
        self.training_config_ = dict(epochs=epochs, seed=seed, lr=lr, weight_decay=weight_decay,
                                     device=str(device), torch_version=str(torch.__version__),
                                     checkpoint_selection='final_epoch')
        for name in ('labels_', 'clustering_config_'):
            self.__dict__.pop(name, None)
        return self

    @classmethod
    def from_legacy_checkpoint(cls, path, data, *, seed=123, epochs=120, **params):
        """Import trusted local figure weights with explicit inputs and settings.

        No configuration is inferred from labels or scores. Caller must verify
        the checkpoint's provenance and the expected saved embedding. Legacy
        files require pickle loading on old PyTorch; never use untrusted files.
        """
        options = dict(map_location='cpu')
        if 'weights_only' in inspect.signature(torch.load).parameters:
            options['weights_only'] = False  # Explicit trusted legacy import.
        import gzip
        if str(path).endswith('.gz'):
            with gzip.open(path, 'rb') as stream:
                checkpoint = torch.load(stream, **options)
        else:
            checkpoint = torch.load(path, **options)
        state = checkpoint.get('model_state')
        if state is None:
            raise ValueError('Legacy checkpoint has no model_state dictionary.')
        result = cls(**params)
        with torch.random.fork_rng(devices=[]):
            result.model_ = result._make_model(data.shape[1], data.shape[2])
        # These are graph buffers of old wrappers, supplied separately here.
        state = {k: v for k, v in state.items() if k not in
                 ('experiment_mask', 'hypergraph_fusion.fixed_P')}
        result.model_.load_state_dict(state, strict=True)
        result.model_.eval()
        with torch.no_grad():
            out = result.model_(**data._on('cpu'))
        result.embedding_ = out['combined'].numpy().copy()
        result.attention_ = {key: out[name].numpy().copy() for key, name in
                            [('rna', 'intra_rna'), ('msi', 'intra_msi'),
                             ('rna_raw', 'intra_rna_raw'), ('msi_raw', 'intra_msi_raw')]}
        result.input_info_ = dict(shape=list(data.shape), fingerprint=data.fingerprint(),
                                 spot_ids=list(data.spot_ids) if data.spot_ids is not None else None)
        result.history_ = []
        result.training_config_ = dict(seed=seed, epochs=epochs, origin='imported_figure_checkpoint',
                                       source=str(path), checkpoint_selection='final_epoch')
        return result

    def _require_fit(self):
        if not hasattr(self, 'model_'):
            raise RuntimeError('Fit or load a model first.')

    def fit_transform(self, data, **kwargs):
        """Fit one model and return its unnormalized final fused representation."""
        return self.fit(data, **kwargs).embedding_.copy()

    def transform(self, data):
        """Replay inference on the same prepared inputs, checking order and graphs.

        This model is transductive; this method does not promise prediction on
        new animals or altered graphs. New inputs require a separate fit.
        """
        self._require_fit()
        if not isinstance(data, PreparedData) or data.fingerprint() != self.input_info_['fingerprint']:
            raise ValueError('Input values, row order or graphs differ from the fitted data.')
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            result = self.model_(**data._on(device))['combined']
        return result.cpu().numpy().copy()

    def cluster(self, n_clusters, *, seed=0, n_init=10):
        """KMeans on row-L2-normalized final embeddings; separate clustering seed."""
        self._require_fit()
        _positive_int('n_clusters', n_clusters)
        _positive_int('n_init', n_init)
        _seed(seed)
        if n_clusters > len(self.embedding_):
            raise ValueError('n_clusters cannot exceed the number of spots.')
        z = F.normalize(torch.from_numpy(self.embedding_), dim=1).numpy()
        self.labels_ = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(z)
        self.clustering_config_ = dict(n_clusters=n_clusters, seed=seed, n_init=n_init,
                                       row_l2=True)
        return self.labels_.copy()

    def save(self, path):
        """Save inference weights, configuration, input hash and results (no optimizer)."""
        self._require_fit()
        checkpoint = dict(format='shine-omics.simple.v1', params=self.get_params(),
                          model_config=self.model_.get_config(), input_info=self.input_info_,
                          training_config=self.training_config_, history=self.history_,
                          state_dict={k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()},
                          embedding=torch.from_numpy(self.embedding_.copy()),
                          attention={k: torch.from_numpy(v.copy()) for k, v in self.attention_.items()})
        if hasattr(self, 'labels_'):
            checkpoint['labels'] = torch.from_numpy(self.labels_.copy())
            checkpoint['clustering_config'] = self.clustering_config_
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(checkpoint, temporary)
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        """Load a checkpoint made by save(), on CPU. Only load trusted files.

        Newer torch uses weights_only=True; older supported torch has pickle
        loading only. A checkpoint is for inference, not resuming optimization.
        """
        options = dict(map_location='cpu')
        if 'weights_only' in inspect.signature(torch.load).parameters:
            options['weights_only'] = True
        checkpoint = torch.load(path, **options)
        if checkpoint.get('format') != 'shine-omics.simple.v1':
            raise ValueError('Not a SHINE simple-API checkpoint; use the matching legacy loader.')
        result = cls(**checkpoint['params'])
        config = checkpoint['model_config']
        with torch.random.fork_rng(devices=[]):
            result.model_ = result._make_model(config['rna_dim'], config['msi_dim'])
        current_config = result.model_.get_config()
        if config.get('schema_version') == 3:
            config = dict(config, schema_version=4, attention_backend='sparse',
                          evaluate_zero_alignment=False, cross_attention_weights='sparse')
        if current_config != config:
            raise ValueError('Saved model configuration differs from this implementation.')
        result.model_.load_state_dict(checkpoint['state_dict'], strict=True)
        result.model_.eval()
        result.embedding_ = checkpoint['embedding'].numpy().copy()
        result.attention_ = {k: v.numpy().copy() for k, v in checkpoint['attention'].items()}
        result.history_ = checkpoint['history']
        result.input_info_ = checkpoint['input_info']
        result.training_config_ = checkpoint['training_config']
        if 'labels' in checkpoint:
            result.labels_ = checkpoint['labels'].numpy().copy()
            result.clustering_config_ = checkpoint['clustering_config']
        return result
