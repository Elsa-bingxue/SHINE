import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from shine import SHINE, prepare_paired, normalize_counts
from shine.revision import RevisionSHINE

torch.set_num_threads(1)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(27)
        self.rna = rng.uniform(.1, 4, (15, 8)).astype('float32')
        self.msi = rng.uniform(.1, 4, (15, 9)).astype('float32')
        self.xy = rng.rand(15, 2).astype('float32')
        self.data, self.graph = prepare_paired(self.rna, self.msi, self.xy,
                spatial_k=3, feature_k=4, hyperedge_sizes=(3, 5), pca_components=4)

    def test_normalization_uses_unselected_genes_for_library_size(self):
        x = np.array([[1, 3, 6], [2, 2, 16]], dtype='float32')
        actual = normalize_counts(x, target_sum=10, feature_indices=[0, 1])
        expected = np.log1p(np.array([[1, 3], [1, 1]], dtype='float32'))
        np.testing.assert_allclose(actual, expected)
        with self.assertRaises(ValueError): normalize_counts([[0, 0]])
        with self.assertRaises(ValueError): normalize_counts([[-1, 2]])

    def test_incidence_and_normalized_propagation_equation(self):
        h = self.graph['incidence'].toarray().astype('float64')
        np.testing.assert_array_equal(h.sum(0), [3]*15+[5]*15)
        dv = np.diag((h.sum(1)+1e-6)**-.5)
        de = np.diag((h.sum(0)+1e-6)**-1)
        expected = dv @ h @ de @ h.T @ dv
        np.testing.assert_allclose(self.graph['propagation'].numpy(), expected, atol=2e-7)
        np.testing.assert_array_equal(self.graph['causal_mask'].sum(1).numpy(), np.repeat(4, 15))
        # Labels are not parameters of construction; deterministic inputs give
        # the same complete data/graph fingerprint.
        rebuilt, _ = prepare_paired(self.rna, self.msi, self.xy, spatial_k=3,
                 feature_k=4, hyperedge_sizes=(3, 5), pca_components=4)
        self.assertEqual(self.data.fingerprint(), rebuilt.fingerprint())

    def test_attention_backends_forward_and_gradient_agree(self):
        outputs, gradients = [], []
        for backend in ['dense', 'sparse', 'grouped']:
            torch.manual_seed(11)
            model = RevisionSHINE(8, 9, latent=8, attention_backend=backend,
                                   normalize_fusion=True, spatial_floor=.5)
            model.eval()
            out = model(**self.data._tensors)
            total = model.loss_terms(out, self.data._tensors['rna'], self.data._tensors['msi'], self.data._tensors['spatial_graph'])['total']
            total.backward()
            outputs.append(out['combined'].detach().numpy())
            gradients.append(model.hypergraph_fusion.weight.grad.numpy())
        for i in [1, 2]:
            np.testing.assert_allclose(outputs[0], outputs[i], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(gradients[0], gradients[i], atol=2e-6, rtol=2e-5)

    def test_grouped_checkpoint_and_legacy_import(self):
        fit = SHINE(loss='reconstruction', latent_dim=8, attention_backend='grouped',
                    evaluate_zero_alignment=True).fit(self.data, epochs=2, seed=123)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'fit.pt';fit.save(path)
            restored = SHINE.load(path)
            np.testing.assert_array_equal(fit.embedding_, restored.transform(self.data))
            legacy = Path(d)/'legacy.pt'
            torch.save({'model_state':fit.model_.state_dict()}, legacy)
            imported = SHINE.from_legacy_checkpoint(legacy, self.data, **fit.get_params())
            np.testing.assert_array_equal(fit.embedding_, imported.embedding_)

    def test_preserves_old_dev0_checkpoint_defaults(self):
        fit = SHINE(latent_dim=8).fit(self.data, epochs=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'old.pt';fit.save(path)
            old = torch.load(path)
            for key in ('attention_backend','evaluate_zero_alignment'):
                old['params'].pop(key);old['model_config'].pop(key)
            old['model_config'].update(schema_version=3,
              cross_attention_weights='Sparse edge order from nonzero causal_mask entries')
            torch.save(old,path)
            np.testing.assert_array_equal(fit.embedding_,SHINE.load(path).transform(self.data))


if __name__ == '__main__': unittest.main()
