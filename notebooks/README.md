# SHINE section notebooks

One executed notebook per section. Default mode replays the saved figure model;
`RUN_NEW_TRAINING=True` starts a separate 120-epoch fit.

| Notebook | Main figure | K | Loss | Pre-fusion L2 | Spatial floor |
|---|---|---:|---|---|---:|
| [01_A1_SHINE.ipynb](01_A1_SHINE.ipynb) | 2 | 11 | Original | On | 0 |
| [02_B1_SHINE.ipynb](02_B1_SHINE.ipynb) | 2 | 11 | Original | On | 0 |
| [03_C1_SHINE.ipynb](03_C1_SHINE.ipynb) | 2 | 11 | Original | Off | 0 |
| [04_mPD3_SN_SHINE.ipynb](04_mPD3_SN_SHINE.ipynb) | 3 | 11 | Original | On | 0 |
| [05_mPD4_SN_SHINE.ipynb](05_mPD4_SN_SHINE.ipynb) | 3 | 11 | Original | On | 0 |
| [06_LC091_SHINE.ipynb](06_LC091_SHINE.ipynb) | 4 | 5 | Reconstruction | Off | 0.5 |
| [07_BC_515_S1_SHINE.ipynb](07_BC_515_S1_SHINE.ipynb) | 5 | 7 | Reconstruction | Off | 0 |

Cross-modal support is self + 4 in A1/B1/C1 and self + 6 elsewhere.
The spatial graph is kNN6 in every section.

## Setup

```bash
python -m pip install "shine-omics[analysis]==0.1.1" jupyterlab
```

Version 0.1.1 includes the independent local-mask option. Copy the supplied
`Updated_Figure_Data` overlay into the existing data root for A1/B1/C1.

The figure-data archive is awaiting public release. Once it is available,
extract it into the repository’s `data/` directory. Alternatively, set
`SHINE_DATA_ROOT` to its extraction directory. See [data requirements](../data/README.md).
The manifest verifies inputs, checkpoints and labels by SHA-256. These files
are separate from the Python package.

## Results

Spatial maps use training seed 123. Five-fit summaries use seeds 0, 1, 2, 123,
1234; KMeans seeds 0–4 are averaged within each fit before computing sample SD.
Displayed clustering uses seed 123 for nigra and 0 for other sections.

Inputs are frozen, preprocessed paired matrices. A1/B1/C1 use the pre-registered SMA data released with [NicheTrans](https://github.com/YSTLab/NicheTrans); mPD3_SN/mPD4_SN use the original SMA author pairing. Registration is upstream.
Molecular maps show measured values, and spot-level comparisons are descriptive.

Saved-model and fresh-training results are reported separately. See
[reproducibility](../shine-omics-pypi/docs/reproducibility.md) for numerical details.

Original code is retained in [legacy](../legacy/README.md).

Spatial panels use `figure_plotting.py`: tissue-clipped nearest-spot cells for Figure 2, native Visium cells for Figure 3, and hexagonal spot footprints for Figures 4–5. Display cells do not add observations or change scores. Grey denotes unlabelled or unpaired tissue; white in a molecular color scale denotes intermediate intensity.
