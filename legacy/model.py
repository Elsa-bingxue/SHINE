import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2, dropout=0.0):
        super(GraphTransformerLayer, self).__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Q, K, V projections
        self.query_lin = nn.Linear(in_dim, out_dim)
        self.key_lin = nn.Linear(in_dim, out_dim)
        self.value_lin = nn.Linear(in_dim, out_dim)

        self.out_lin = nn.Linear(out_dim, out_dim)

        # projection for residual connection (newly added!)
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, adj):
        residual = self.residual_proj(x)  # Match dimensions

        Q = self.query_lin(x).view(-1, self.num_heads, self.head_dim)
        K = self.key_lin(x).view(-1, self.num_heads, self.head_dim)
        V = self.value_lin(x).view(-1, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("ihd,jhd->ijh", Q, K) / (self.head_dim ** 0.5)

        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj

        attn_scores = attn_scores.masked_fill(adj_dense.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        h_prime = torch.einsum("ijh,jhd->ihd", attn_weights, V)
        h_prime = h_prime.reshape(-1, self.num_heads * self.head_dim)

        h_prime = self.out_lin(h_prime)
        h_prime = self.norm(h_prime + residual)  # First Add & Norm

        # Feed-Forward Network
        ffn_output = self.ffn(h_prime)
        h_prime = self.norm(ffn_output + h_prime)  # Second Add & Norm

        return h_prime

class SpatialGCNEncoder(Module):
    def __init__(self, in_feat, out_feat, hidden_dim=64, dropout=0.2):
        super(SpatialGCNEncoder, self).__init__()

        self.feature_linear = nn.Linear(in_feat, hidden_dim)
        self.coord_linear = nn.Linear(2, hidden_dim)
        self.gcn_weight = Parameter(torch.FloatTensor(hidden_dim, out_feat))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_linear.weight)
        nn.init.xavier_uniform_(self.coord_linear.weight)
        nn.init.xavier_uniform_(self.gcn_weight)

    def forward(self, features, coordinates, adj):
        feat_emb = self.activation(self.feature_linear(features))
        coord_emb = self.activation(self.coord_linear(coordinates))
        x = feat_emb + coord_emb
        x = self.norm(x)
        x = self.dropout(x)
        x = torch.spmm(adj, x)
        x = torch.mm(x, self.gcn_weight)
        x = self.activation(x)
        return x

class Encoder_overall(Module):
    def __init__(self, dim_in_feat_omics1, dim_latent_feat_omics1,
                 dim_in_feat_omics2, dim_latent_feat_omics2,
                 dim_output_recon1, dim_output_recon2,
                 dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()

        # Encoder latent dimensions
        self.feature_encoder1 = GraphTransformerLayer(dim_in_feat_omics1, dim_latent_feat_omics1)
        self.feature_encoder2 = GraphTransformerLayer(dim_in_feat_omics2, dim_latent_feat_omics2)

        self.spatial_encoder1 = SpatialGCNEncoder(dim_in_feat_omics1, dim_latent_feat_omics1)
        self.spatial_encoder2 = SpatialGCNEncoder(dim_in_feat_omics2, dim_latent_feat_omics2)

        # Decoder output dimensions adjusted to original feature dimensions
        self.decoder_omics1 = Decoder(dim_latent_feat_omics1, dim_output_recon1)
        self.decoder_omics2 = Decoder(dim_latent_feat_omics2, dim_output_recon2)

        # Attention layers
        self.atten_omics1 = AttentionLayer(dim_latent_feat_omics1, dim_latent_feat_omics1)
        self.atten_omics2 = AttentionLayer(dim_latent_feat_omics2, dim_latent_feat_omics2)


        # Hypergraph fusion
        self.hypergraph_fusion = HypergraphFusionLayer(
            dim_latent_feat_omics1 + dim_latent_feat_omics2,
            dim_latent_feat_omics1)

        # Cross-modal attention layer
        self.cross_modal_attention = CrossModalAttentionLayer(dim_latent_feat_omics1, dim_latent_feat_omics2)

        self.causal_attention = CausalAttentionLayer(
            dim_latent_feat_omics1, dim_latent_feat_omics2, dropout=dropout
        )

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, features_omics1, features_omics2,
                coordinates_omics1, coordinates_omics2,
                adj_feature_omics1,adj_feature_omics2,
                adj_spatial_omics1, adj_spatial_omics2, H):

        emb_spatial_omics1 = self.spatial_encoder1(features_omics1, coordinates_omics1, adj_spatial_omics1)
        emb_spatial_omics2 = self.spatial_encoder2(features_omics2, coordinates_omics2, adj_spatial_omics2)


        emb_feature_omics1 = self.feature_encoder1(features_omics1, adj_feature_omics1)
        emb_feature_omics2 = self.feature_encoder2(features_omics2, adj_feature_omics2)

        emb_latent_omics1, attn_weights_omics1 = self.atten_omics1(
            emb_spatial_omics1, emb_feature_omics1
        )
        emb_latent_omics2, attn_weights_omics2 = self.atten_omics2(
            emb_spatial_omics2, emb_feature_omics2
        )

        causal_mask = torch.ones((emb_latent_omics1.size(0), emb_latent_omics2.size(0)),
                                 device=emb_latent_omics1.device)

        attended_emb, causal_attn_weights = self.causal_attention(
            emb_latent_omics1, emb_latent_omics2, causal_mask
        )
        emb_latent_combined_causal = torch.cat([emb_latent_omics1, attended_emb], dim=1)
        emb_hypergraph_fused = self.hypergraph_fusion(emb_latent_combined_causal, H)

        emb_recon_omics1 = self.decoder_omics1(emb_hypergraph_fused, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_hypergraph_fused, adj_spatial_omics2)

        emb_latent_omics1_mapped = torch.sparse.mm(adj_spatial_omics2, emb_latent_omics1)
        emb_latent_omics1_across_recon = self.feature_encoder2(
            self.decoder_omics2(emb_latent_omics1_mapped, adj_spatial_omics2), adj_spatial_omics2
        )

        emb_latent_omics2_mapped = torch.sparse.mm(adj_spatial_omics1, emb_latent_omics2)
        emb_latent_omics2_across_recon = self.feature_encoder1(
            self.decoder_omics1(emb_latent_omics2_mapped, adj_spatial_omics1), adj_spatial_omics1
        )

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_hypergraph_fused,
            'attended_emb': attended_emb,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,

        }
        return results

class Encoder(Module):
    """
    Modality-specific GNN encoder.
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)  # Linear transformation
        x = torch.spmm(adj, x)  # Graph message passing using adjacency matrix
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class Decoder(Module):
    """
    Modality-specific GNN decoder.
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)  # Linear transformation
        x = torch.spmm(adj, x)  # Graph message passing using adjacency matrix
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class AttentionLayer(Module):
    """
    Improved Attention layer for intra-modality integration.
    """

    def __init__(self, in_feat, out_feat, dropout=0.0):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Learnable parameters
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))  # Attention transformation
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))       # Attention weights projection

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        """
        Forward pass for intra-modality attention.

        Args:
            emb1 (torch.Tensor): Embedding for spatial graph.
            emb2 (torch.Tensor): Embedding for feature graph.

        Returns:
            emb_combined (torch.Tensor): Combined embedding.
            alpha (torch.Tensor): Attention weights.
        """
        emb = torch.stack([emb1, emb2], dim=1)  # Shape: [N, 2, in_feat]

        # Compute attention scores
        v = torch.tanh(torch.matmul(emb, self.w_omega))  # Shape: [N, 2, out_feat]
        vu = torch.matmul(v, self.u_omega)              # Shape: [N, 2, 1]
        alpha = F.softmax(vu.squeeze(-1), dim=1)        # Shape: [N, 2]

        # Combine embeddings using attention weights
        emb_combined = torch.sum(emb * alpha.unsqueeze(-1), dim=1)  # Shape: [N, out_feat]

        return emb_combined, alpha

class CrossModalAttentionLayer(Module):
    """
    Cross-modal attention layer for integrating two modalities.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0):
        super(CrossModalAttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Learnable parameters for attention mechanism
        self.query_weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.key_weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.value_weight = Parameter(torch.FloatTensor(in_feat, out_feat))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query_weight)
        torch.nn.init.xavier_uniform_(self.key_weight)
        torch.nn.init.xavier_uniform_(self.value_weight)

    def forward(self, emb1, emb2):
        """
        Args:
            emb1 (torch.Tensor): Embedding for modality 1 [N, in_feat].
            emb2 (torch.Tensor): Embedding for modality 2 [N, in_feat].

        Returns:
            emb_combined (torch.Tensor): Combined embedding [N, out_feat*2].
            attended_emb (torch.Tensor): Modality 2 embedding attended by modality 1 [N, out_feat].
            num_nodes_omics1 (int): Number of nodes in modality 1.
            num_nodes_omics2 (int): Number of nodes in modality 2.
        """
        # Compute query from modality 1, key/value from modality 2
        query = torch.matmul(emb1, self.query_weight)  # [N, out_feat]
        key = torch.matmul(emb2, self.key_weight)      # [N, out_feat]
        value = torch.matmul(emb2, self.value_weight)  # [N, out_feat]

        # Attention scores [N, N], each node in emb1 attends to nodes in emb2
        attention_scores = torch.matmul(query, key.T) / (self.out_feat ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [N, N]

        # Apply attention weights to value (modality 2)
        attended_emb = torch.matmul(attention_weights, value)  # [N, out_feat]

        # Cross-modal fusion by concatenating original emb1 and attended emb2
        emb_combined = torch.cat([query, attended_emb], dim=1)  # [N, out_feat*2]

        return emb_combined, attended_emb, emb1.size(0), emb2.size(0)

class CausalAttentionLayer(Module):

    def __init__(self, in_feat, out_feat, dropout=0.0):
        super(CausalAttentionLayer, self).__init__()
        self.query = nn.Linear(in_feat, out_feat)
        self.key = nn.Linear(in_feat, out_feat)
        self.value = nn.Linear(in_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb_cause, emb_effect, causal_mask):
        Q = self.query(emb_cause)
        K = self.key(emb_effect)
        V = self.value(emb_effect)

        attn_scores = torch.matmul(Q, K.T) / (Q.size(-1)**0.5)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        causal_output = torch.matmul(attn_weights, V)
        return causal_output, attn_weights

class HypergraphFusionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0):
        super(HypergraphFusionLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, emb_cat, H):
        Dv = torch.diag(torch.sum(H, dim=1))
        De = torch.diag(torch.sum(H, dim=0))

        Dv_inv_sqrt = torch.inverse(torch.sqrt(Dv + torch.eye(Dv.size(0)).to(Dv.device)*1e-6))
        De_inv = torch.inverse(De + torch.eye(De.size(0)).to(De.device)*1e-6)

        H_T = H.T
        X = emb_cat @ self.weight

        emb_hypergraph = Dv_inv_sqrt @ H @ De_inv @ H_T @ Dv_inv_sqrt @ X
        emb_hypergraph = F.relu(emb_hypergraph)
        emb_hypergraph = self.dropout(emb_hypergraph)

        return emb_hypergraph
