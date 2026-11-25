# SHINE section notebooks

One executed notebook per section. Default mode replays the saved figure model;
`RUN_NEW_TRAINING=True` starts a separate 120-epoch fit.

| Notebook | Main figure | K | Loss | Pre-fusion L2 | Spatial floor |
|---|---|---:|---|---|---:|
| [01_A1_SHINE.ipynb](01_A1_SHINE.ipynb) | 2 | 11 | Original revision | On | 0 |
| [02_B1_SHINE.ipynb](02_B1_SHINE.ipynb) | 2 | 11 | Original revision | On | 0 |
| [03_C1_SHINE.ipynb](03_C1_SHINE.ipynb) | 2 | 11 | Original revision | Off | 0 |
| [04_mPD3_SN_SHINE.ipynb](04_mPD3_SN_SHINE.ipynb) | 3 | 11 | Original revision | On | 0 |
| [05_mPD4_SN_SHINE.ipynb](05_mPD4_SN_SHINE.ipynb) | 3 | 11 | Original revision | On | 0 |
| [06_LC091_SHINE.ipynb](06_LC091_SHINE.ipynb) | 4 | 5 | Reconstruction | Off | 0.5 |
| [07_BC_515_S1_SHINE.ipynb](07_BC_515_S1_SHINE.ipynb) | 5 | 7 | Reconstruction | Off | 0 |

## Setup

```bash
python -m pip install "shine-omics[analysis]==0.1.0" jupyterlab
```

Download the [figure-data archive](https://github.com/Elsa-bingxue/SHINE/releases/tag/v0.1.0)
and extract it into the repository’s `data/` directory. Alternatively, set
`SHINE_DATA_ROOT` to its extraction directory. See [data requirements](../data/README.md).
The manifest verifies inputs, checkpoints and labels by SHA-256. These files
are separate from the Python package.

## Results

Spatial maps use training seed 123. Five-fit summaries use seeds 0, 1, 2, 123,
1234; KMeans seeds 0–4 are averaged within each fit before computing sample SD.
Displayed clustering uses seed 123 for nigra and 0 for other sections.

Inputs are frozen, preprocessed paired matrices. Registration is upstream.
Molecular maps show measured values, and spot-level comparisons are descriptive.

All seven saved-model replays matched the figure embeddings and labels. Seven
fresh 120-epoch fits reproduced six partitions. mPD4 differed by about 2e-6 in
its embedding but selected a different KMeans partition; both results remain
in its notebook. The 14 five-fit endpoint summaries match the figure tables.

Original code is retained in [legacy](../legacy/README.md).
