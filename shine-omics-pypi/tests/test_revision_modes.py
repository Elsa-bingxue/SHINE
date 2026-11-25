"""Regression checks against the frozen experiment implementation, not new fits."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

from shine.revision import RevisionSHINE, standardize_teacher_targets

REFERENCE = os.environ.get('SHINE_REFERENCE_EXPERIMENT')
if REFERENCE:
    sys.path.insert(0, str(Path(REFERENCE)))
    import model_ops as reference
else:
    reference = None

torch.set_num_threads(1)


class TrainingModesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.x, self.m = torch.rand(24, 8), torch.rand(24, 9)
        self.xy = torch.rand(24, 2)
        dense = torch.eye(24)
        dense += torch.roll(dense, 1, 0) + torch.roll(dense, -1, 0)
        self.adj = (dense / dense.sum(1, keepdim=True)).to_sparse().coalesce()
        self.mask = dense.ne(0).float()
        self.p = torch.eye(24) * .6 + torch.ones(24, 24) / 24 * .4
        raw = (torch.randn(24, 128), torch.randn(24, 128))
        self.targets, _ = standardize_teacher_targets(*raw)

    def run_model(self, model):
        out = model(self.x, self.m, self.xy, self.adj, self.adj, self.adj,
                    causal_mask=self.mask, propagation=self.p)
        targets = self.targets if model.training_mode == 'retention' else None
        return out, model.loss_terms(out, self.x, self.m, self.adj, teacher_targets=targets)

    def make_reference(self, mode):
        torch.manual_seed(123)
        old = reference.Encoder_overall(8, 128, 9, 128, 8, 9)
        reference.extend(old, mode)
        old.register_buffer('experiment_mask', self.mask)
        old.hypergraph_fusion.register_buffer('fixed_P', self.p)
        import types
        old.hypergraph_fusion.forward = types.MethodType(reference.fast_hyper, old.hypergraph_fusion)
        return old

    def reference_step(self, model):
        out = reference.forward(model, self.x, self.m, self.xy, self.adj, self.adj, self.adj)
        loss = reference.losses(model, out, self.x, self.m,
                                self.targets if model.arm == 'retention' else None)
        if model.arm == 'original':
            z1, z2 = out['rna'], out['msi']
            c1 = model.feature_encoder2(model.decoder_omics2(torch.sparse.mm(self.adj, z1), self.adj), self.adj)
            c2 = model.feature_encoder1(model.decoder_omics1(torch.sparse.mm(self.adj, z2), self.adj), self.adj)
            sim = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / .07
            ids = torch.arange(len(z1))
            loss['total'] = loss['total'] + F.mse_loss(z1, c1) + F.mse_loss(z2, c2) + .025 * (F.cross_entropy(sim, ids) + F.cross_entropy(sim.T, ids))
        return out, loss

    @unittest.skipIf(reference is None, 'Set SHINE_REFERENCE_EXPERIMENT for frozen-run regression.')
    def test_reference_forward_gradients_and_optimizer(self):
        for mode in ('original', 'reconstruction_only', 'retention', 'dgl'):
            with self.subTest(mode=mode):
                old = self.make_reference(mode)
                torch.manual_seed(123)
                new = RevisionSHINE(8, 9, training_mode=mode)
                for name, value in new.state_dict().items():
                    self.assertTrue(torch.equal(value, old.state_dict()[name]), name)
                opts = [torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-3) for m in (old, new)]
                for step in range(3):
                    old.train(); new.train()
                    for opt in opts:opt.zero_grad()
                    torch.manual_seed(200 + step);a, la = self.reference_step(old)
                    torch.manual_seed(200 + step);b, lb = self.run_model(new)
                    self.assertTrue(torch.allclose(a['combined'], b['combined'], atol=3e-5, rtol=2e-4))
                    self.assertTrue(torch.allclose(la['total'], lb['total'], atol=3e-5, rtol=2e-4))
                    la['total'].backward();lb['total'].backward()
                    for (na, pa), (nb, pb) in zip(old.named_parameters(), new.named_parameters()):
                        self.assertEqual(na, nb)
                        self.assertEqual(pa.grad is None, pb.grad is None, na)
                        if pa.grad is not None:
                            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=5e-5, rtol=2e-3), na)
                    for opt in opts:opt.step()

    def test_dgl_gradient_separation(self):
        model = RevisionSHINE(8, 9, training_mode='dgl')
        out, loss = self.run_model(model)
        enc = list(model.feature_encoder1.parameters()) + list(model.spatial_encoder1.parameters())
        fused = list(model.hypergraph_fusion.parameters()) + list(model.causal_attention.parameters())
        def norm(term, pars):
            grads = torch.autograd.grad(term, pars, allow_unused=True, retain_graph=True)
            return sum(float(v.abs().sum()) for v in grads if v is not None)
        self.assertEqual(norm(loss['fused_reconstruction'], enc), 0)
        self.assertEqual(norm(loss['unimodal_reconstruction'], fused), 0)
        self.assertGreater(norm(loss['unimodal_reconstruction'], enc), 0)
        self.assertGreater(norm(loss['fused_reconstruction'], fused), 0)

    def test_l2_each_mode_and_final_relu(self):
        for mode in ('original', 'reconstruction_only', 'retention', 'dgl'):
            with self.subTest(mode=mode):
                model = RevisionSHINE(8, 9, training_mode=mode, normalize_fusion=True)
                out, loss = self.run_model(model)
                for part in out['concat'].chunk(2, dim=1):
                    self.assertTrue(torch.allclose(part.norm(dim=1), torch.ones(24), atol=1e-6))
                self.assertTrue((out['combined'] >= 0).all())
                loss['total'].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_teacher_frozen_population_standardization(self):
        rna = torch.randn(24, 128, requires_grad=True)
        msi = torch.ones(24, 128, requires_grad=True)
        targets, stats = standardize_teacher_targets(rna, msi)
        self.assertFalse(any(t.requires_grad for t in targets))
        self.assertTrue(torch.allclose(targets[0].std(0, unbiased=False), torch.ones(128), atol=1e-6))
        self.assertTrue(torch.equal(targets[1], torch.zeros_like(msi)))
        model = RevisionSHINE(8, 9, training_mode='retention')
        out, loss = self.run_model(model)
        with self.assertRaises(ValueError):model.loss_terms(out, self.x, self.m, self.adj)
        with self.assertRaises(ValueError):model.loss_terms(out, self.x, self.m, self.adj, teacher_targets=(rna, msi))

    def test_zero_alignment_matches_reconstruction_training(self):
        for l2 in (False, True):
            with self.subTest(normalize_fusion=l2):
                torch.manual_seed(123)
                old = RevisionSHINE(8, 9, training_mode='reconstruction_only', normalize_fusion=l2)
                torch.manual_seed(123)
                new = RevisionSHINE(8, 9, normalize_fusion=l2,
                                    correspondence_weight=0, infonce_weight=0)
                self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                opts = [torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-3)
                        for m in (old, new)]
                for _ in range(3):
                    for opt in opts:opt.zero_grad()
                    rng = torch.get_rng_state()
                    a, la = self.run_model(old)
                    rng_after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    b, lb = self.run_model(new)
                    self.assertTrue(torch.equal(rng_after, torch.get_rng_state()))
                    self.assertEqual(set(la), set(lb))
                    self.assertNotIn('InfoNCE', lb)
                    self.assertNotIn('RNA_correspondence', lb)
                    self.assertTrue(torch.equal(a['combined'], b['combined']))
                    self.assertTrue(torch.equal(la['total'], lb['total']))
                    la['total'].backward();lb['total'].backward()
                    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
                        self.assertEqual(p.grad is None, q.grad is None, name)
                        if p.grad is not None:self.assertTrue(torch.equal(p.grad, q.grad), name)
                    for opt in opts:opt.step()
                    for name, value in old.state_dict().items():
                        self.assertTrue(torch.equal(value, new.state_dict()[name]), name)

    def test_independent_alignment_weights_and_gradients(self):
        for corr, infonce in [(1., 0.), (0., .05), (.3, .02)]:
            with self.subTest(correspondence_weight=corr, infonce_weight=infonce):
                full = RevisionSHINE(8, 9).eval()
                selected = RevisionSHINE(8, 9, correspondence_weight=corr,
                                         infonce_weight=infonce).eval()
                selected.load_state_dict(full.state_dict(), strict=True)
                _, full_terms = self.run_model(full)
                _, terms = self.run_model(selected)
                expected = (full_terms['fused_reconstruction']
                            + corr * full_terms['RNA_correspondence']
                            + corr * full_terms['MSI_correspondence']
                            + infonce * full_terms['InfoNCE'])
                self.assertTrue(torch.allclose(expected, terms['total'], atol=1e-6, rtol=1e-6))
                self.assertEqual('RNA_correspondence' in terms, corr > 0)
                self.assertEqual('InfoNCE' in terms, infonce > 0)
                expected.backward();terms['total'].backward()
                for (name, p), (_, q) in zip(full.named_parameters(), selected.named_parameters()):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5), name)

    def test_loss_weight_configuration_roundtrip(self):
        for weight_name in ('correspondence_weight', 'infonce_weight'):
            for invalid in (-1, float('nan'), float('inf'), '0', True):
                with self.assertRaises(ValueError):RevisionSHINE(8, 9, **{weight_name:invalid})
            for mode in ('reconstruction_only', 'retention', 'dgl'):
                with self.assertRaises(ValueError):
                    RevisionSHINE(8, 9, training_mode=mode, **{weight_name:.1})
        for corr, infonce in [(0., 0.), (1., 0.), (0., .05), (.3, .02)]:
            model = RevisionSHINE(8, 9, correspondence_weight=corr, infonce_weight=infonce).eval()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'model.pt'
                torch.save({'model_state':model.state_dict(), 'config':model.get_config()}, path)
                saved = torch.load(path, map_location='cpu')
            keys = ('rna_dim', 'msi_dim', 'latent', 'training_mode', 'normalize_fusion',
                    'retention_weight', 'correspondence_weight', 'infonce_weight')
            clone = RevisionSHINE(**{k:saved['config'][k] for k in keys}).eval()
            clone.load_state_dict(saved['model_state'], strict=True)
            self.assertEqual(model.get_config(), clone.get_config())
            self.assertEqual(saved['config']['loss_weights']['rna_correspondence'], corr)
            self.assertEqual(saved['config']['loss_weights']['contrastive'], infonce)
            a, la = self.run_model(model);b, lb = self.run_model(clone)
            self.assertTrue(torch.equal(a['combined'], b['combined']))
            self.assertTrue(torch.equal(la['total'], lb['total']))

    def test_invalid_configuration_and_checkpoint_roundtrip(self):
        with self.assertRaises(ValueError):RevisionSHINE(8, 9, training_mode='best')
        with self.assertRaises(TypeError):RevisionSHINE(8, 9, normalize_fusion='false')
        for mode in ('original', 'reconstruction_only', 'retention', 'dgl'):
            model = RevisionSHINE(8, 9, training_mode=mode).eval()
            before, _ = self.run_model(model)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.pt'
                torch.save({'model_state':model.state_dict(), 'config':model.get_config()}, path)
                checkpoint=torch.load(path, map_location='cpu')
            clone=RevisionSHINE(8, 9, training_mode=mode).eval()
            clone.load_state_dict(checkpoint['model_state'], strict=True)
            after, _ = self.run_model(clone)
            self.assertTrue(torch.equal(before['combined'], after['combined']))
            self.assertEqual(model.get_config(), clone.get_config())
            if mode != 'original':
                self.assertEqual(model.get_config()['loss_weights']['rna_correspondence'], 0.)
                self.assertEqual(model.get_config()['loss_weights']['contrastive'], 0.)

    def test_spatial_floor_validation_and_unchanged_initialization(self):
        for invalid in (-.01, 1.01, float('nan'), float('inf'), '0.5', True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RevisionSHINE(8, 9, spatial_floor=invalid)
        torch.manual_seed(123)
        base = RevisionSHINE(8, 9)
        for rho in (0., .5, 1.):
            torch.manual_seed(123)
            model = RevisionSHINE(8, 9, spatial_floor=rho)
            self.assertEqual(set(base.state_dict()), set(model.state_dict()))
            for name, value in base.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_zero_spatial_floor_preserves_default_and_rng(self):
        torch.manual_seed(123)
        base = RevisionSHINE(8, 9)
        torch.manual_seed(123)
        explicit = RevisionSHINE(8, 9, spatial_floor=0.)
        rng = torch.get_rng_state()
        a, la = self.run_model(base)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        b, lb = self.run_model(explicit)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertTrue(torch.equal(a['combined'], b['combined']))
        self.assertTrue(torch.equal(la['total'], lb['total']))
        for mod in ('rna', 'msi'):
            self.assertTrue(torch.equal(a['intra_' + mod], a['intra_' + mod + '_raw']))
        la['total'].backward(); lb['total'].backward()
        for (name, p), (_, q) in zip(base.named_parameters(), explicit.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:self.assertTrue(torch.equal(p.grad, q.grad), name)

    def test_spatial_floor_bounds_and_training_modes(self):
        for mode in ('original', 'reconstruction_only', 'retention', 'dgl'):
            for l2 in (False, True):
                with self.subTest(mode=mode, normalize_fusion=l2):
                    model = RevisionSHINE(8, 9, training_mode=mode,
                                          normalize_fusion=l2, spatial_floor=.5)
                    out, terms = self.run_model(model)
                    for mod in ('rna', 'msi'):
                        w = out['intra_' + mod]
                        self.assertTrue((w[:, 0] >= .5).all())
                        self.assertTrue((w[:, 0] <= 1).all())
                        self.assertTrue(torch.allclose(w.sum(1), torch.ones(24)))
                        self.assertFalse(torch.equal(w, out['intra_' + mod + '_raw']))
                    self.assertTrue((out['combined'] >= 0).all())
                    terms['total'].backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None))
                    self.assertGreater(float(model.atten_omics1.u_omega.grad.norm()), 0)
                    self.assertGreater(float(model.atten_omics2.u_omega.grad.norm()), 0)

    def test_unit_floor_only_uses_spatial_branch_at_mixing(self):
        model = RevisionSHINE(8, 9, training_mode='reconstruction_only', spatial_floor=1.)
        out, loss = self.run_model(model)
        self.assertTrue(torch.equal(out['rna'], out['spatial_encoder1']))
        self.assertTrue(torch.equal(out['msi'], out['spatial_encoder2']))
        loss['total'].backward()
        for feature in (model.feature_encoder1, model.feature_encoder2):
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                                for p in feature.parameters()))
        self.assertGreater(float(model.spatial_encoder1.gcn_weight.grad.norm()), 0)

    def test_nonzero_floor_checkpoint_roundtrip(self):
        model = RevisionSHINE(8, 9, spatial_floor=.5, normalize_fusion=True).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save({'model_state':model.state_dict(), 'config':model.get_config()}, path)
            saved = torch.load(path, map_location='cpu')
        config = saved['config']
        keys = ('rna_dim', 'msi_dim', 'latent', 'training_mode', 'normalize_fusion',
                'retention_weight', 'correspondence_weight', 'infonce_weight')
        clone = RevisionSHINE(**{k:config[k] for k in keys},
                              spatial_floor=config.get('spatial_floor', 0.)).eval()
        clone.load_state_dict(saved['model_state'], strict=True)
        a, la = self.run_model(model); b, lb = self.run_model(clone)
        self.assertEqual(config['spatial_floor'], .5)
        self.assertEqual(model.get_config(), clone.get_config())
        self.assertTrue(torch.equal(a['combined'], b['combined']))
        self.assertTrue(torch.equal(la['total'], lb['total']))


if __name__ == '__main__':unittest.main(verbosity=2)
