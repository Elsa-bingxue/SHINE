# Model and implementation

The current API is `from shine import SHINE`. It calls
[`RevisionSHINE`](../shine-omics-pypi/shine/revision.py), with graph construction
in [`pipeline.py`](../shine-omics-pypi/shine/pipeline.py).
The older `shine.model` and `shine.Training` entry points are retained for
historical analyses and do not define the model used in Figures 2–5.

## Graphs

Spatial kNN6 support is symmetrized and given self-connections. The normalization
routine adds another identity, so the spatial diagonal has weight two before
normalization. Modality-specific feature kNN20 graphs use correlation distance.
For either supplied adjacency $A$, the implemented operator is

$$\widetilde A=D^{-1/2}(A+I)^\mathsf{T}D^{-1/2},\qquad D_{ii}=\sum_j(A+I)_{ij}.$$

Feature attention uses its nonzero support. The transpose matters for the
directed feature graph; spatial support is symmetric.

Each modality contributes 30 principal components, divided by the Frobenius
norm of its score matrix. The concatenated scores define two sets of kNN
neighborhoods, of sizes 5 and 20 including the centre. For row-oriented
membership matrices $B_k$, the node-by-hyperedge incidence is
$H=[B_5^\mathsf{T},B_{20}^\mathsf{T}]$.

## Attention and fusion

Each modality has a spatial GCN and a two-head feature Transformer. A softmax
combines the two branch embeddings at each spot. For spatial floor $\rho$,

$$\alpha_{sp}=\rho+(1-\rho)a_{sp},\qquad \alpha_{feat}=(1-\rho)a_{feat}.$$

RNA embeddings provide queries; MSI embeddings provide keys and values, with
learned affine projections. Cross-modal attention uses the same spot and `local_k` other nearest spatial
neighbors, independently of the spatial encoder graph. The default is self+6;
Figure 2 uses self+4 and Figures 3–5 use self+6. Setting `local_mask=False`
retains learned attention over all paired spots. The name `causal_attention` is retained
in checkpoint keys; this is spatial masking and does not imply causal inference.

The RNA embedding and attended MSI embedding are concatenated. With `l2=True`,
each is normalized separately immediately before concatenation. Fusion is

$$A_H=D_v^{-1/2}HD_e^{-1}H^\mathsf{T}D_v^{-1/2},\qquad Z_H=\operatorname{ReLU}(A_HZ_CW_H).$$

Degrees include a $10^{-6}$ offset. Fusion dropout is zero in the main figure
settings. The fixed incidence-derived operator is equivalent to a GCN supplied
with $A_H$; it has no learned intermediate hyperedge state.

## Reconstruction and losses

Two GCN decoders reconstruct the modalities from $Z_H$. The correspondence
paths reuse the opposite modality's decoder and feature Transformer, comparing
the returned latent embedding with the starting embedding. Contrastive loss is
symmetric InfoNCE on row-normalized **pre-fusion** modality embeddings, with
temperature 0.07 and no additional projection head.

$$L=2L_{RNA}^{rec}+2L_{MSI}^{rec}+\lambda_{corr}(L_{RNA}^{corr}+L_{MSI}^{corr})+\lambda_{CL}L^{CL}.$$

`loss="original"` uses $(\lambda_{corr},\lambda_{CL})=(1,0.05)$.
`loss="reconstruction"` uses $(0,0)$. MSE averages all matrix entries.
Training uses Adam, 120 epochs, learning rate $10^{-4}$ and weight decay
$10^{-3}$. Pre-fusion L2 normalization and optimizer weight decay are separate.

The backbone retains several unused legacy modules for checkpoint compatibility.
The active forward pass and loss are `RevisionSHINE.forward` and
`RevisionSHINE.loss_terms`; independent post-fusion Transformer reconstruction
heads and projection heads are not used in the main configurations.

See [reproducibility](../shine-omics-pypi/docs/reproducibility.md) for numerical
settings and the distinction between saved-model replay and fresh training.
