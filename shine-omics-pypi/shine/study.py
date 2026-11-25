"""Reproducible, per-section notebook support; no experiment-code imports.

The JSON manifest supplies relative source paths and hashes. Labels are used
only for evaluation and descriptive molecular comparisons, never training.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .api import SHINE
from .pipeline import prepare_paired


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


class Section:
    """One frozen figure cohort with auditable paths, parameters and results."""
    def __init__(self, root, manifest, case):
        self.root = Path(root)
        self.manifest = json.loads(Path(manifest).read_text())
        self.case = case
        self.config = self.manifest['cases'][case]
        self.spots = pd.read_csv(self.path('spots'))
        self.features = json.loads(self.path('features').read_text())
        for key in ('inputs', 'spots', 'features'):
            self.verify(key)
        with np.load(self.path('inputs'), allow_pickle=False) as source:
            self.rna, self.msi, self.xy = [source[k].copy() for k in ('rna', 'msi', 'xy')]
        if (len(self.spots) != self.config['n_spots'] or not self.spots.spot.is_unique
                or not np.allclose(self.xy, self.spots[['x', 'y']].to_numpy())):
            raise ValueError('Frozen cohort or coordinate order has changed.')
        if self.rna.shape != (len(self.spots), len(self.features['genes'])) or self.msi.shape != (len(self.spots), len(self.features['msi'])):
            raise ValueError('Feature order does not match the input matrices.')

    def path(self, name):
        return self.root / self.config['sources'][name]['path']

    def verify(self, name):
        expected = self.config['sources'][name]['sha256']
        if file_sha256(self.path(name)) != expected:
            raise ValueError('Source changed since figure manifest was frozen: ' + name)

    def prepare(self):
        return prepare_paired(self.rna, self.msi, self.xy, spot_ids=self.spots.spot,
                              **self.config['graph_options'])

    def replay(self, data):
        """Recompute seed123 from saved weights and verify its saved embedding."""
        self.verify('checkpoint_123')
        self.verify('embedding_123')
        model = SHINE.from_legacy_checkpoint(self.path('checkpoint_123'), data,
                    seed=123, epochs=120, **self.config['model_options'])
        replay = torch.nn.functional.normalize(torch.from_numpy(model.embedding_), dim=1).numpy()
        saved = np.load(self.path('embedding_123'), allow_pickle=False)
        error = float(np.max(np.abs(replay - saved)))
        if not np.allclose(replay, saved, atol=3e-5, rtol=3e-4):
            raise AssertionError('Saved-model replay mismatch: ' + str(error))
        # Canonical labels use the frozen normalized embedding. This avoids
        # hiding floating point ties behind a new cluster-ID permutation.
        labels = KMeans(self.config['K'], n_init=10,
                        random_state=self.config['display_clustering_seed']).fit_predict(saved)
        reference = self.figure_labels()
        if not np.array_equal(labels, reference):
            raise AssertionError('Recomputed seed123 labels differ from the figure.')
        return model, labels, dict(embedding_max_abs_error=error,
                                  figure_labels_exact=True, n_spots=len(labels),
                                  origin='saved figure checkpoint; new package inference')

    def figure_labels(self):
        self.verify('display_labels')
        f = pd.read_csv(self.path('display_labels'))
        if not f.spot.equals(self.spots.spot):
            raise ValueError('Figure labels have a different spot order.')
        return f[self.config['label_column']].to_numpy()

    def endpoints(self):
        s = self.spots
        if self.config['kind'] == 'cancer':
            return {'Whole section': (s.used_for_region_metrics.to_numpy(bool), s.region.fillna('Unlabelled').to_numpy())}
        ref = s.annotation_region_lesion.fillna('Unlabelled').copy()
        if self.config['kind'] == 'striatum':
            ref.loc[s.region.eq('ACB')] = 'ACB'
        out = {'Whole section': (s.used_for_region_lesion_metrics.to_numpy(bool), ref.to_numpy())}
        for region in (['CP'] if self.config['kind'] == 'striatum' else ['SNr', 'VTA-SNc']):
            mask = (s.region.eq(region) & s.lesion.isin(['intact', 'lesioned'])).to_numpy()
            out[region] = mask, s.lesion.fillna('Unlabelled').to_numpy()
        return out

    def score(self, labels):
        if len(labels) != len(self.spots):
            raise ValueError('Partition has a different cohort.')
        return pd.DataFrame([dict(endpoint=name, n_scored=int(mask.sum()),
                                  ARI=adjusted_rand_score(truth[mask], labels[mask]),
                                  NMI=normalized_mutual_info_score(truth[mask], labels[mask]))
                             for name, (mask, truth) in self.endpoints().items()])

    def five_fit_metrics(self):
        """Five training fits, averaging KMeans seeds 0..4 inside each fit.

        Sample SD is over five training seeds, not over 25 nested partitions.
        No region-specific reclustering is performed.
        """
        rows = []
        for seed in self.manifest['training_seeds']:
            self.verify('embedding_' + str(seed))
            z = np.load(self.path('embedding_' + str(seed)), allow_pickle=False)
            for cs in self.manifest['clustering_seeds']:
                labels = KMeans(self.config['K'], n_init=10, random_state=cs).fit_predict(z)
                table = self.score(labels)
                table['training_seed'], table['clustering_seed'] = seed, cs
                rows.append(table)
        raw = pd.concat(rows, ignore_index=True)
        per_fit = raw.groupby(['endpoint', 'training_seed'], as_index=False)[['ARI', 'NMI']].mean()
        summary = per_fit.groupby('endpoint')[['ARI', 'NMI']].agg(['mean', 'std'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary['n_fits'] = per_fit.groupby('endpoint').size()
        return raw, per_fit, summary.reset_index()

    def molecular_values(self):
        """Measured values of current figure markers, in the same spot order.

        Brain gene panels may include genes outside the model's 2000 inputs;
        these are explicitly read from measured RNA, not imputed by SHINE.
        """
        cfg = self.config
        genes, peaks = cfg['display_genes'], cfg['display_peaks']
        if cfg['kind'] == 'nigra':
            self.verify('measured_molecules')
            table = pd.read_csv(self.path('measured_molecules'))
            table = table[table.animal.eq(cfg['animal'])].set_index('spot')
            if not table.index.is_unique or not self.spots.spot.isin(table.index).all():
                raise ValueError('Missing or duplicated molecular spots.')
            return table.loc[self.spots.spot, genes + peaks].reset_index(drop=True)
        names = self.features['genes']
        if cfg['kind'] == 'striatum':
            self.verify('raw_rna')
            rna = read_10x_genes(self.path('raw_rna'), self.spots.spot, genes)
        else:
            rna = pd.DataFrame(self.rna[:, [names.index(g) for g in genes]], columns=genes)
        mz = np.asarray([float(str(v).replace('mz_', '')) for v in self.features['msi']])
        for peak in peaks:
            j = np.flatnonzero(np.abs(mz - float(peak.replace('mz_', ''))) < 5.1e-6)
            if len(j) != 1:
                raise ValueError('MSI feature match is ambiguous: ' + peak)
            rna[peak] = self.msi[:, j[0]]
        return rna

    def molecular_comparison(self, values, labels):
        """Descriptive effects; no spot-as-independent p values are reported."""
        groups = []
        if self.config['kind'] == 'cancer':
            focus = self.config['molecular_cluster']
            groups = [('C%d versus rest' % focus, labels != focus, labels == focus)]
        else:
            for region in (['CP'] if self.config['kind'] == 'striatum' else ['SNr', 'VTA-SNc']):
                base = self.spots.region.eq(region)
                groups.append((region + ': lesioned minus intact',
                               (base & self.spots.lesion.eq('intact')).to_numpy(),
                               (base & self.spots.lesion.eq('lesioned')).to_numpy()))
        rows = []
        for name, lower, upper in groups:
            for feature in values:
                a = values[feature].to_numpy()
                sd = a[lower | upper].std(ddof=1)
                difference = a[upper].mean() - a[lower].mean()
                rows.append(dict(contrast=name, feature=feature, n_reference=int(lower.sum()),
                                 n_target=int(upper.sum()), mean_reference=a[lower].mean(),
                                 mean_target=a[upper].mean(), difference_SD=difference/sd if sd > 0 else np.nan))
        return pd.DataFrame(rows)


def read_10x_genes(path, spots, genes):
    """Measured RNA: sum duplicate symbols, all-gene normalization, then subset."""
    import h5py
    with h5py.File(path, 'r') as f:
        m = f['matrix']
        barcodes = pd.Index(m['barcodes'][:].astype(str))
        positions = barcodes.get_indexer(spots)
        if not barcodes.is_unique or (positions < 0).any():
            raise ValueError('RNA barcodes do not match the frozen cohort.')
        x = sparse.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:]))[:, positions]
        keep = m['features/feature_type'][:].astype(str) == 'Gene Expression'
        symbols = m['features/name'][:].astype(str)[keep]
    names, codes = np.unique(symbols, return_inverse=True)
    combine = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(names), len(codes)))
    x = combine @ x[keep]
    total = np.asarray(x.sum(0)).ravel()
    if (total <= 0).any():
        raise ValueError('Zero RNA library size.')
    x = x.multiply(10000 / total).tocsr()
    x.data = np.log1p(x.data)
    indices = pd.Index(names).get_indexer(genes)
    if (indices < 0).any():
        raise ValueError('A requested gene was not measured.')
    return pd.DataFrame(x[indices].toarray().T, columns=genes)
