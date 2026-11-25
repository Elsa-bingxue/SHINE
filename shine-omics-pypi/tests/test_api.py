"""End-to-end checks of the wrapper against a direct revision training loop."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from shine import SHINE, prepare
from shine.revision import RevisionSHINE

torch.set_num_threads(1)


class SimpleAPITest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        n = 12
        edge = torch.eye(n) + torch.eye(n).roll(1, 0) + torch.eye(n).roll(-1, 0)
        self.inputs = dict(rna=torch.rand(n, 8), msi=torch.rand(n, 9),
                           coordinates=torch.rand(n, 2),
                           feature_graph_rna=edge.to_sparse(),
                           feature_graph_msi=edge.to_sparse(),
                           spatial_graph=(edge / 3).to_sparse(), causal_mask=edge,
                           propagation=.5 * torch.eye(n) + .5 * torch.ones(n, n) / n)
        self.data = prepare(**self.inputs, spot_ids=['spot' + str(i) for i in range(n)])

    def test_direct_training_equivalence_and_rng(self):
        for loss, l2, floor in [('original', True, 0.), ('reconstruction', False, .5)]:
            with self.subTest(loss=loss):
                torch.manual_seed(33)
                before = torch.random.get_rng_state().clone()
                fit = SHINE(loss=loss, l2=l2, spatial_floor=floor, latent_dim=8)
                fit.fit(self.data, epochs=3, seed=123)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                torch.manual_seed(123)
                direct = RevisionSHINE(8, 9, latent=8, normalize_fusion=l2,
                                       spatial_floor=floor,
                                       training_mode='original' if loss == 'original' else 'reconstruction_only')
                opt = torch.optim.Adam(direct.parameters(), lr=1e-4, weight_decay=1e-3)
                for row in fit.history_:
                    direct.train()
                    opt.zero_grad(set_to_none=True)
                    out = direct(**self.inputs)
                    terms = direct.loss_terms(out, self.inputs['rna'], self.inputs['msi'],
                                              self.inputs['spatial_graph'])
                    self.assertEqual(float(terms['total']), row['total'])
                    terms['total'].backward()
                    opt.step()
                direct.eval()
                with torch.no_grad():
                    expected = direct(**self.inputs)
                np.testing.assert_array_equal(expected['combined'].numpy(), fit.embedding_)
                for name, tensor in direct.state_dict().items():
                    self.assertTrue(torch.equal(tensor, fit.model_.state_dict()[name]), name)
                np.testing.assert_array_equal(expected['intra_rna'].numpy(), fit.attention_['rna'])
                self.assertGreaterEqual(fit.embedding_.min(), 0)
                self.assertGreaterEqual(fit.attention_['rna'][:, 0].min(), floor)

    def test_save_load_replays_exact_embedding_and_labels(self):
        fit = SHINE(loss='reconstruction', l2=True, spatial_floor=.5, latent_dim=8)
        z = fit.fit_transform(self.data, epochs=2, seed=123)
        labels = fit.cluster(3, seed=2)
        expected = KMeans(n_clusters=3, n_init=10, random_state=2).fit_predict(
            F.normalize(torch.from_numpy(z), dim=1).numpy())
        np.testing.assert_array_equal(expected, labels)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            fit.save(path)
            restored = SHINE.load(path)
            np.testing.assert_array_equal(z, restored.transform(self.data))
            np.testing.assert_array_equal(z, restored.embedding_)
            np.testing.assert_array_equal(labels, restored.labels_)
            self.assertEqual(fit.training_config_, restored.training_config_)
            self.assertEqual(fit.input_info_, restored.input_info_)
            self.assertEqual(fit.model_.get_config(), restored.model_.get_config())
            self.assertEqual(fit.history_, restored.history_)

    def test_changed_inputs_order_or_graph_rejected_on_replay(self):
        fit = SHINE(loss='reconstruction', latent_dim=8).fit(self.data, epochs=1)
        for name in ('rna', 'propagation'):
            changed = {k: v.clone() for k, v in self.inputs.items()}
            changed[name][0, 0] += .01
            data = prepare(**changed, spot_ids=self.data.spot_ids)
            with self.assertRaisesRegex(ValueError, 'differ'):
                fit.transform(data)
        data = prepare(**self.inputs, spot_ids=list(reversed(self.data.spot_ids)))
        with self.assertRaisesRegex(ValueError, 'differ'):
            fit.transform(data)

    def test_prepare_validation_and_numpy_scipy_graphs(self):
        import scipy.sparse as sp
        converted = {k: sp.coo_matrix(v.to_dense().numpy()) if v.is_sparse else v.numpy()
                     for k, v in self.inputs.items()}
        data = prepare(**converted, spot_ids=self.data.spot_ids)
        self.assertEqual(self.data.fingerprint(), data.fingerprint())
        fit = SHINE(loss='reconstruction', latent_dim=8).fit(data, epochs=1)
        dense = {k: v.to_dense() if v.is_sparse else v for k, v in self.inputs.items()}
        SHINE(latent_dim=8).fit(prepare(**dense), epochs=1)
        self.assertEqual(fit.embedding_.shape, (12, 8))
        for key, value in [('msi', torch.rand(11, 9)),
                           ('rna', torch.full((12, 8), float('nan'))),
                           ('feature_graph_rna', torch.zeros(12, 12))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare(**dict(self.inputs, **{key: value}))
        with self.assertRaises(ValueError):
            prepare(**self.inputs, spot_ids=['same'] * 12)

    def test_invalid_settings_and_no_implicit_fit(self):
        for settings in [dict(loss='dgl'), dict(l2='false'), dict(spatial_floor=1.1),
                         dict(latent_dim=7), dict(loss='reconstruction', infonce_weight=.1)]:
            with self.subTest(settings=settings), self.assertRaises((ValueError, TypeError)):
                SHINE(**settings)
        fit = SHINE(latent_dim=8)
        with self.assertRaises(RuntimeError):
            fit.cluster(3)
        for kwargs in [dict(epochs=0), dict(seed=-1), dict(lr=0), dict(weight_decay=-1)]:
            with self.assertRaises(ValueError):
                fit.fit(self.data, **kwargs)

    def test_refit_clears_labels_and_fit_transform_is_a_copy(self):
        fit = SHINE(loss='reconstruction', latent_dim=8)
        returned = fit.fit_transform(self.data, epochs=1, seed=0)
        returned[:] = -1
        self.assertGreaterEqual(fit.embedding_.min(), 0)
        fit.cluster(3)
        fit.fit(self.data, epochs=1, seed=123)
        self.assertFalse(hasattr(fit, 'labels_'))
        self.assertEqual(len(fit.history_), 1)


if __name__ == '__main__':
    unittest.main()
