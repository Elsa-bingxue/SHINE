"""Small synthetic example; install shine-omics before running."""
import numpy as np
from shine import SHINE, normalize_counts, prepare_paired

rng = np.random.RandomState(123)
n = 36
rna = normalize_counts(rng.poisson(3, (n, 40)))
msi = normalize_counts(rng.gamma(2, 1, (n, 32)))
xy = rng.rand(n, 2)

data, graphs = prepare_paired(rna, msi, xy,
                              spot_ids=["spot_" + str(i) for i in range(n)])
model = SHINE(loss="reconstruction", l2=False, spatial_floor=0.5)
embedding = model.fit_transform(data, epochs=3, seed=123)  # Full runs: 120.
labels = model.cluster(n_clusters=3, seed=0)
print("Embedding:", embedding.shape, "| Labels:", labels.shape)
