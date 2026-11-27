# Data for SHINE

## Original data

| Sections | Source | Download |
|---|---|---|
| A1 / mPD1, B1 / mPD3, C1 / mPD4 striatum (FMP-10) | Vicari et al., 2024 | [Mendeley Data](https://data.mendeley.com/datasets/w7nw4km7xd/1); RNA section IDs `V11L12-109_A1`, `_B1`, `_C1` |
| mPD3 and mPD4 substantia nigra (FMP-10) | Vicari et al., 2024 | [Mendeley Data](https://data.mendeley.com/datasets/w7nw4km7xd/1) |
| LC091 lung cancer | Godfrey et al., 2025 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZFCWC), `LC_091.h5ad` |
| BC_515_S1 breast cancer | Godfrey et al., 2025 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZFCWC), `BC_515_Section_1.h5ad` |

The earlier DHB example and cell-type workflows remain in
[legacy](../legacy/README.md). The mouse annotation archive is
[Zeisel_stsc_output.csv.zip](../legacy/Mouse%20striatum%20%28FMP-10%29/Zeisel_stsc_output.csv.zip).
The earlier breast cell-type analysis used
[GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078).

For the striatum (A1/B1/C1), the notebooks use pre-registered SMA data released with [NicheTrans](https://github.com/YSTLab/NicheTrans), rather than rerunning registration inside SHINE. Substantia nigra uses the original SMA author pairing.

## Frozen inputs for the seven notebooks

`SHINE_figure_data_v0.1.0.tar.gz` is prepared and validated, but has not yet been
released publicly. Version 0.1.1 also requires the supplied `Updated_Figure_Data`
overlay for the four-neighbor A1/B1/C1 fits. Copy its contents into the same
extraction directory before verification. The original-study downloads above
remain available.
The archive contains the exact paired input matrices, coordinates, feature and
spot order, seed-123 weights, embeddings from five fits, reference labels, and
molecular tables and tissue display geometry required by the notebooks. It is a reproduction subset, not
a replacement for the full original datasets. Source-study attribution and
original data terms continue to apply.

| Section | Paired spots | Pairing used in the figure |
|---|---:|---|
| A1 | 2,204 | NicheTrans pre-registered SMA data; frozen common cohort |
| B1 | 2,455 | NicheTrans pre-registered SMA data; frozen common cohort |
| C1 | 2,378 | NicheTrans pre-registered SMA data; frozen common cohort |
| mPD3_SN | 2,915 | Author SMA pairing |
| mPD4_SN | 3,021 | Author SMA pairing |
| LC091 | 2,141 | Frozen original paired matrices |
| BC_515_S1 | 3,508 | Frozen original paired matrices |

From the repository root:

```bash
tar -xzf SHINE_figure_data_v0.1.0.tar.gz -C data
python scripts/check_figure_data.py --root data
```

The extracted directory contains `revision_experiments/`, `outputs/`, `MSI&ST/`,
and `SHINE-github/`. Keep their relative paths. For a different location, set
`SHINE_DATA_ROOT` to the extraction directory. The notebooks find their manifest
next to the notebooks or in the archive’s `SHINE-github/notebooks/` directory.

`inputs.npz` stores `rna`, `msi`, and `xy` in matching spot order; `spots.csv`
contains annotations and evaluation masks, and `feature_order.json` fixes the
column order. Do not normalize these matrices again. Annotation labels enter
evaluation and descriptive downstream analysis only. Not all paired spots
have reference labels, so the number scored can be smaller than the number
used for training.

[replay_files.json](replay_files.json) records the current reproduction files and
their SHA-256 values, including the updated striatal files.
[SHA256SUMS](SHA256SUMS) verifies the earlier compressed base archive. Raw image
registration is upstream and is not repeated by the seven section notebooks.
The archive provides SHINE reproduction inputs; it does not include every
external benchmark run or regenerate the full composite manuscript figures.

## References

1. Vicari, M. et al. Spatial multimodal analysis of transcriptomes and metabolomes
   in tissues. *Nature Biotechnology* 42, 1046–1050 (2024).
   https://doi.org/10.1038/s41587-023-01937-y
2. Godfrey, T. M. et al. Integrating ambient ionization mass spectrometry imaging
   and spatial transcriptomics on the same cancer tissues to identify
   RNA–metabolite correlations. *Angewandte Chemie International Edition* 64,
   e202502028 (2025). https://doi.org/10.1002/anie.202502028
3. Wu, S. Z. et al. A single-cell and spatially resolved atlas of human breast
   cancers. *Nature Genetics* 53, 1334–1347 (2021).
   https://doi.org/10.1038/s41588-021-00911-1
