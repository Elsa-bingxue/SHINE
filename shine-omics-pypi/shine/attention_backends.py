"""Numerical execution backends for reproducing the frozen figure runs.

All implement the same masked attention. They are recorded because floating
point summation order can affect a long stochastic training trajectory.
"""
import types
import torch


def _groups(layer, adj):
    if getattr(layer, '_group_graph', None) is adj:
        return layer._group_support
    if adj.is_sparse:
        q = adj.coalesce()
        idx = q.indices()[:, q.values().ne(0)]
    else:
        idx = adj.ne(0).nonzero(as_tuple=False).T
    counts = torch.bincount(idx[0], minlength=adj.shape[0])
    if counts.min() <= 0:
        raise ValueError('Every attention row needs neighbors.')
    offsets = torch.cumsum(counts, 0) - counts
    pos = torch.arange(idx.shape[1], device=idx.device) - offsets[idx[0]]
    groups, lower, upper = [], 0, 1
    while lower < int(counts.max()):
        rows = ((counts > lower) & (counts <= upper)).nonzero(as_tuple=False).flatten()
        if len(rows):
            local = torch.full((adj.shape[0],), -1, dtype=torch.long, device=idx.device)
            local[rows] = torch.arange(len(rows), device=idx.device)
            selected = local[idx[0]] >= 0
            cols = torch.zeros((len(rows), int(counts[rows].max())), dtype=torch.long, device=idx.device)
            valid = torch.zeros_like(cols, dtype=torch.bool)
            cols[local[idx[0, selected]], pos[selected]] = idx[1, selected]
            valid[local[idx[0, selected]], pos[selected]] = True
            groups.append((rows, cols, valid))
        lower, upper = upper, upper * 2
    layer._group_graph, layer._group_support = adj, groups
    return groups


def _feature(self, x, adj):
    q = self.query_lin(x).view(-1, self.num_heads, self.head_dim)
    k = self.key_lin(x).view(-1, self.num_heads, self.head_dim)
    v = self.value_lin(x).view(-1, self.num_heads, self.head_dim)
    h = torch.zeros_like(q)
    for rows, cols, valid in _groups(self, adj):
        scores = (q[rows, None] * k[cols]).sum(-1) / self.head_dim ** .5
        weights = self.dropout(torch.softmax(scores.masked_fill(~valid[:, :, None], float('-inf')), dim=1))
        h = h.index_copy(0, rows, (weights[:, :, :, None] * v[cols]).sum(1))
    h = self.norm(self.out_lin(h.reshape(len(x), -1)) + self.residual_proj(x))
    return self.norm(self.ffn(h) + h)


def _cross(self, x, m, mask):
    q, k, v = self.query(x), self.key(m), self.value(m)
    out, weights_all = torch.zeros_like(v), []
    for rows, cols, valid in _groups(self, mask):
        scores = (q[rows, None] * k[cols]).sum(-1) / q.shape[-1] ** .5
        weights = self.dropout(torch.softmax(scores.masked_fill(~valid, float('-inf')), dim=1))
        out = out.index_copy(0, rows, (weights[:, :, None] * v[cols]).sum(1))
        weights_all.append(weights)
    return out, weights_all


def install_grouped(model, *, cross=True):
    for layer in (model.feature_encoder1, model.feature_encoder2):
        layer.forward = types.MethodType(_feature, layer)
    if cross:
        model.causal_attention.forward = types.MethodType(_cross, model.causal_attention)
