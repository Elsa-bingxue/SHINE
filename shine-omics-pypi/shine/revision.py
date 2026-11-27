"""Configurable objectives for the tested H5+20/local-attention SHINE revision.

Graphs and matched input rows are supplied by the caller. This module does not
choose a mode, K, checkpoint or seed from tissue labels or evaluation scores.
The older ``shine.model`` and 300-epoch ``shine.Training`` entry remain separate.
"""
import copy
import math
import types

import torch
import torch.nn.functional as F

from .revision_backbone import Encoder_overall

TRAINING_MODES = ('original', 'reconstruction_only', 'retention', 'dgl')


def _neighbors(adj):
    if adj.is_sparse:
        adj = adj.coalesce()
        idx = adj.indices()[:, adj.values().ne(0)]
    else:
        idx = torch.nonzero(adj.ne(0), as_tuple=False).T
    if torch.unique(idx[0]).numel() != adj.shape[0]:
        raise ValueError('Every attention row needs at least one neighbor.')
    return idx


def _edge_softmax(scores, idx, n):
    sparse = torch.sparse_coo_tensor(idx, scores, (n, n, scores.shape[1])).coalesce()
    if not torch.equal(sparse.indices(), idx):
        raise ValueError('Attention edge indices must be sorted and unique.')
    return torch.sparse.softmax(sparse, dim=1).coalesce().values()


def _feature_attention(self, x, adj):
    idx = _neighbors(adj)
    row, col = idx
    q = self.query_lin(x).view(-1, self.num_heads, self.head_dim)
    k = self.key_lin(x).view(-1, self.num_heads, self.head_dim)
    v = self.value_lin(x).view(-1, self.num_heads, self.head_dim)
    scores = (q[row] * k[col]).sum(-1) / (self.head_dim ** .5)
    weights = self.dropout(_edge_softmax(scores, idx, len(x)))
    h = torch.zeros_like(q).index_add(0, row, weights[:, :, None] * v[col])
    h = self.norm(self.out_lin(h.reshape(len(x), -1)) + self.residual_proj(x))
    return self.norm(self.ffn(h) + h)


def _cross_attention(self, x, m, mask):
    idx = _neighbors(mask)
    row, col = idx
    q, k, v = self.query(x), self.key(m), self.value(m)
    scores = (q[row] * k[col]).sum(-1, keepdim=True) / (q.shape[-1] ** .5)
    weights = self.dropout(_edge_softmax(scores, idx, len(x)))
    attended = torch.zeros_like(v).index_add(0, row, weights * v[col])
    return attended, weights


def standardize_teacher_targets(rna, msi, sd_floor=1e-4):
    """Freeze same-row teacher latents using the tested population-SD convention.

    The caller must verify matching spot order, features and training seed.
    Compute these once from the final teacher evaluation, never from the
    student's current minibatch. The reference retention runs use a DGL-style
    teacher whose encoders receive only unimodal reconstruction gradients.
    """
    if not math.isfinite(sd_floor) or sd_floor <= 0:
        raise ValueError('sd_floor must be positive and finite.')
    if rna.ndim != 2 or msi.shape != rna.shape:
        raise ValueError('Teacher latents must have identical [spots, latent] shapes.')
    targets, stats = [], {}
    for name, value in [('rna', rna), ('msi', msi)]:
        if not torch.isfinite(value).all():
            raise ValueError('Teacher latents must be finite.')
        value = value.detach()
        mean = value.mean(0)
        sd = value.std(0, unbiased=False).clamp_min(sd_floor)
        targets.append(((value - mean) / sd).detach())
        stats[name + '_mean'], stats[name + '_sd'] = mean, sd
    return tuple(targets), stats


class RevisionSHINE(Encoder_overall):
    """Same inference backbone, explicit optional training strategies.

    ``original`` is the fixed-loss 120-epoch revision baseline, not the older
    300-epoch dynamically weighted objective. Its correspondence and InfoNCE
    weights can be independently set to zero, without changing the backbone.
    ``retention`` and ``dgl`` add
    training-only heads; retention also needs a separately fitted frozen teacher.
    ``normalize_fusion`` is pre-concatenation L2 normalization, not weight decay.
    ``spatial_floor`` bounds the spatial branch's mixing weight separately in
    each modality. Zero preserves the original learned attention exactly.
    """
    def __init__(self, rna_dim, msi_dim, latent=128, *, training_mode='original',
                 normalize_fusion=False, retention_weight=.1,
                 correspondence_weight=None, infonce_weight=None,
                 spatial_floor=0.0, attention_backend='sparse',
                 evaluate_zero_alignment=False):
        if attention_backend not in ('sparse', 'dense', 'grouped', 'grouped_feature'):
            raise ValueError('Unknown attention execution backend.')
        if type(evaluate_zero_alignment) is not bool:
            raise TypeError('evaluate_zero_alignment must be bool.')
        if training_mode not in TRAINING_MODES:
            raise ValueError('training_mode must be one of ' + repr(TRAINING_MODES))
        if type(normalize_fusion) is not bool:
            raise TypeError('normalize_fusion must be a bool.')
        try:
            valid_floor = (not isinstance(spatial_floor, bool)
                           and math.isfinite(spatial_floor)
                           and 0 <= spatial_floor <= 1)
        except TypeError:
            valid_floor = False
        if not valid_floor:
            raise ValueError('spatial_floor must be a finite number between 0 and 1.')
        if not math.isfinite(retention_weight) or retention_weight < 0:
            raise ValueError('retention_weight must be nonnegative and finite.')
        weights = {}
        for name, value, default in (
                ('correspondence_weight', correspondence_weight, 1.),
                ('infonce_weight', infonce_weight, .05)):
            if value is None:
                value = default if training_mode == 'original' else 0.
            try:
                valid = not isinstance(value, bool) and math.isfinite(value) and value >= 0
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(name + ' must be a nonnegative finite number.')
            weights[name] = float(value)
        if training_mode != 'original' and any(weights.values()):
            raise ValueError('Nonzero alignment weights require training_mode="original"; '
                             'the other modes retain their tested objectives.')
        if any(type(v) is not int or v <= 0 for v in (rna_dim, msi_dim, latent)) or latent % 2:
            raise ValueError('Feature dimensions must be positive integers; latent must be even.')
        super().__init__(rna_dim, latent, msi_dim, latent, rna_dim, msi_dim)
        self.training_mode = training_mode
        self.normalize_fusion = normalize_fusion
        self.spatial_floor = float(spatial_floor)
        self.retention_weight = float(retention_weight)
        self.correspondence_weight = weights['correspondence_weight']
        self.infonce_weight = weights['infonce_weight']
        self.rna_dim, self.msi_dim, self.latent = rna_dim, msi_dim, latent
        self.attention_backend = attention_backend
        self.evaluate_zero_alignment = evaluate_zero_alignment
        if training_mode == 'dgl':
            self.unimodal_decoder1 = copy.deepcopy(self.decoder_omics1)
            self.unimodal_decoder2 = copy.deepcopy(self.decoder_omics2)
        elif training_mode == 'retention':
            # Preserve the reference backbone initialization and dropout RNG.
            with torch.random.fork_rng(devices=[]):
                self.retain1 = torch.nn.Linear(latent, latent)
                self.retain2 = torch.nn.Linear(latent, latent)
                for head in (self.retain1, self.retain2):
                    torch.nn.init.eye_(head.weight)
                    torch.nn.init.zeros_(head.bias)
        # Edge-wise evaluation preserves the tested masked-attention equations.
        if attention_backend == 'sparse':
            for layer in (self.feature_encoder1, self.feature_encoder2):
                layer.forward = types.MethodType(_feature_attention, layer)
            self.causal_attention.forward = types.MethodType(_cross_attention, self.causal_attention)
        elif attention_backend in ('grouped', 'grouped_feature'):
            from .attention_backends import install_grouped
            install_grouped(self, cross=attention_backend == 'grouped')

    def get_config(self):
        return dict(schema_version=4, rna_dim=self.rna_dim, msi_dim=self.msi_dim,
                    attention_backend=self.attention_backend,
                    evaluate_zero_alignment=self.evaluate_zero_alignment,
                    latent=self.latent, training_mode=self.training_mode,
                    normalize_fusion=self.normalize_fusion,
                    spatial_floor=self.spatial_floor,
                    retention_weight=self.retention_weight,
                    correspondence_weight=self.correspondence_weight,
                    infonce_weight=self.infonce_weight, final_relu=True,
                    loss_weights=dict(rna_reconstruction=2., msi_reconstruction=2.,
                                      rna_correspondence=self.correspondence_weight,
                                      msi_correspondence=self.correspondence_weight,
                                      contrastive=self.infonce_weight, temperature=.07,
                                      unimodal_reconstruction=2. if self.training_mode == 'dgl' else 0.,
                                      retention=self.retention_weight if self.training_mode == 'retention' else 0.),
                    cross_attention_weights=self.attention_backend)

    def _within_modality(self, layer, spatial, feature):
        combined, raw = layer(spatial, feature)
        if self.spatial_floor == 0:
            return combined, raw, raw
        rho = self.spatial_floor
        effective = torch.stack((rho + (1 - rho) * raw[:, 0],
                                 (1 - rho) * raw[:, 1]), dim=1)
        combined = torch.sum(torch.stack((spatial, feature), dim=1)
                             * effective.unsqueeze(-1), dim=1)
        return combined, effective, raw

    def forward(self, rna, msi, coordinates, feature_graph_rna, feature_graph_msi,
                spatial_graph, *, causal_mask, propagation):
        """Use fixed H5+20 propagation and local self+6 mask for existing runs.

        No annotated regions or pathology labels are accepted here. ``propagation``
        is Dv^-1/2 H De^-1 H.T Dv^-1/2, computed once from the supplied inputs.
        """
        n = len(rna)
        if (rna.ndim != 2 or msi.ndim != 2 or len(msi) != n
                or rna.shape[1] != self.rna_dim or msi.shape[1] != self.msi_dim
                or coordinates.shape != (n, 2)):
            raise ValueError('Paired inputs and coordinates have incompatible shapes.')
        if any(a.shape != (n, n) for a in
               (feature_graph_rna, feature_graph_msi, spatial_graph, causal_mask, propagation)):
            raise ValueError('All graphs and propagation must match the paired spot rows.')
        if propagation.requires_grad or causal_mask.requires_grad:
            raise ValueError('The evaluated revision uses fixed input-derived graphs.')
        s1 = self.spatial_encoder1(rna, coordinates, spatial_graph)
        s2 = self.spatial_encoder2(msi, coordinates, spatial_graph)
        f1 = self.feature_encoder1(rna, feature_graph_rna)
        f2 = self.feature_encoder2(msi, feature_graph_msi)
        z1, a1, raw1 = self._within_modality(self.atten_omics1, s1, f1)
        z2, a2, raw2 = self._within_modality(self.atten_omics2, s2, f2)
        q, v = (z1.detach(), z2.detach()) if self.training_mode == 'dgl' else (z1, z2)
        attended, weights = self.causal_attention(q, v, causal_mask)
        branches = [q, attended]
        if self.normalize_fusion:
            branches = [F.normalize(b, dim=1) for b in branches]
        concat = torch.cat(branches, dim=1)
        combined = self.hypergraph_fusion.dropout(F.relu(propagation @ (concat @ self.hypergraph_fusion.weight)))
        out = dict(feature_encoder1=f1, feature_encoder2=f2,
                   spatial_encoder1=s1, spatial_encoder2=s2, rna=z1, msi=z2,
                   intra_rna=a1, intra_msi=a2,
                   intra_rna_raw=raw1, intra_msi_raw=raw2, attended_msi=attended,
                   cross_attention=weights, concat=concat, combined=combined,
                   rna_reconstruction=self.decoder_omics1(combined, spatial_graph),
                   msi_reconstruction=self.decoder_omics2(combined, spatial_graph))
        if self.training_mode == 'dgl':
            out['unimodal_rna_reconstruction'] = self.unimodal_decoder1(z1, spatial_graph)
            out['unimodal_msi_reconstruction'] = self.unimodal_decoder2(z2, spatial_graph)
        return out

    def loss_terms(self, out, rna, msi, spatial_graph, *, teacher_targets=None):
        terms = {'RNA_MSE': F.mse_loss(out['rna_reconstruction'], rna),
                 'MSI_MSE': F.mse_loss(out['msi_reconstruction'], msi)}
        terms['fused_reconstruction'] = 2 * (terms['RNA_MSE'] + terms['MSI_MSE'])
        terms['total'] = terms['fused_reconstruction']
        if self.correspondence_weight > 0 or self.evaluate_zero_alignment:
            z1, z2 = out['rna'], out['msi']
            c1 = self.feature_encoder2(self.decoder_omics2(torch.sparse.mm(spatial_graph, z1), spatial_graph), spatial_graph)
            c2 = self.feature_encoder1(self.decoder_omics1(torch.sparse.mm(spatial_graph, z2), spatial_graph), spatial_graph)
            terms['RNA_correspondence'] = F.mse_loss(z1, c1)
            terms['MSI_correspondence'] = F.mse_loss(z2, c2)
            terms['total'] = (terms['total']
                              + self.correspondence_weight * terms['RNA_correspondence']
                              + self.correspondence_weight * terms['MSI_correspondence'])
        if self.infonce_weight > 0 or self.evaluate_zero_alignment:
            z1, z2 = out['rna'], out['msi']
            sim = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / .07
            target = torch.arange(len(rna), device=rna.device)
            terms['InfoNCE'] = (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target)) / 2
            terms['total'] = terms['total'] + self.infonce_weight * terms['InfoNCE']
        if self.training_mode == 'dgl':
            terms['unimodal_RNA_MSE'] = F.mse_loss(out['unimodal_rna_reconstruction'], rna)
            terms['unimodal_MSI_MSE'] = F.mse_loss(out['unimodal_msi_reconstruction'], msi)
            terms['unimodal_reconstruction'] = 2 * (terms['unimodal_RNA_MSE'] + terms['unimodal_MSI_MSE'])
            terms['total'] = terms['total'] + terms['unimodal_reconstruction']
        elif self.training_mode == 'retention':
            if teacher_targets is None or len(teacher_targets) != 2:
                raise ValueError('Retention requires two frozen, standardized teacher targets.')
            for value in teacher_targets:
                if (value.shape != out['combined'].shape or value.requires_grad
                        or not torch.isfinite(value).all()
                        or value.device != out['combined'].device):
                    raise ValueError('Teacher targets must be finite, frozen and aligned to the student rows/device.')
            terms['retain_RNA_MSE'] = F.mse_loss(self.retain1(out['combined']), teacher_targets[0])
            terms['retain_MSI_MSE'] = F.mse_loss(self.retain2(out['combined']), teacher_targets[1])
            terms['retention'] = self.retention_weight * (terms['retain_RNA_MSE'] + terms['retain_MSI_MSE'])
            terms['total'] = terms['total'] + terms['retention']
        if teacher_targets is not None and self.training_mode != 'retention':
            raise ValueError('Teacher targets are used only in retention mode.')
        return terms
