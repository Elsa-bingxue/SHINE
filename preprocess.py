import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

def convert_edge_index_to_adj(edge_index, num_nodes):
    """
    Converts edge_index to a dense adjacency matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index (2 x num_edges) from PyTorch Geometric Data object.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    scipy.sparse.coo_matrix
        Dense adjacency matrix.
    """
    adj = sp.coo_matrix(
        (np.ones(edge_index.size(1)), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
        shape=(num_nodes, num_nodes)
    )
    return adj

def preprocess_graph(adj):
    """
    Normalizes the adjacency matrix for GNN input.

    Parameters
    ----------
    adj : scipy.sparse.coo_matrix
        Sparse adjacency matrix.

    Returns
    -------
    torch.sparse.FloatTensor
        Normalized adjacency matrix in sparse tensor format.
    """
    adj = sp.coo_matrix(adj)  # Ensure COO format
    adj_ = adj + sp.eye(adj.shape[0])  # Add self-loops
    rowsum = np.array(adj_.sum(1))  # Degree matrix (row sums)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # D^(-1/2)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # A_normalized
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Converts a scipy sparse matrix to a PyTorch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse.coo_matrix
        Sparse matrix.

    Returns
    -------
    torch.sparse.FloatTensor
        Sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # Ensure COO format
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adjacent_matrix_preprocessing(transcript_data, metabolite_data):
    """
    Converts dense adjacency matrices to sparse and normalized adjacency matrices.
    Constructs both spatial and feature graphs for PyTorch Geometric Data objects.

    Parameters
    ----------
    transcript_data : Data
        Transcriptomics graph data (PyTorch Geometric Data object).
    metabolite_data : Data
        Metabolomics graph data (PyTorch Geometric Data object).

    Returns
    -------
    dict
        A dictionary containing preprocessed adjacency matrices for both spatial and feature graphs.
    """

    ######################################## Spatial Graph ########################################
    # Spatial adjacency for transcriptomics
    adj_spatial_omics1 = preprocess_graph(
        convert_edge_index_to_adj(transcript_data.edge_index, transcript_data.num_nodes)
    )

    # Spatial adjacency for metabolomics
    adj_spatial_omics2 = preprocess_graph(
        convert_edge_index_to_adj(metabolite_data.edge_index, metabolite_data.num_nodes)
    )

    ######################################## Feature Graph ########################################
    # For feature graphs, we assume that `x` (features) is available
    adj_feature_omics1 = kneighbors_graph(
        transcript_data.x.cpu().numpy(), n_neighbors=20, mode="connectivity", metric="correlation", include_self=False
    )
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1.toarray())  # Normalize

    adj_feature_omics2 = kneighbors_graph(
        metabolite_data.x.cpu().numpy(), n_neighbors=20, mode="connectivity", metric="correlation", include_self=False
    )
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2.toarray())  # Normalize

    # Pack results into a dictionary
    adj = {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
    }

    return adj

def construct_hypergraph(features_transcript, features_metabolite, n_neighbors=10):
    combined_features = torch.cat([features_transcript, features_metabolite], dim=1).cpu().numpy()

    adj = kneighbors_graph(combined_features, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    H = adj.toarray().astype(np.float32)

    return torch.tensor(H, dtype=torch.float32).to(features_transcript.device)


