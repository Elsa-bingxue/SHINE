# SHINE: Decoding metabolic–transcriptional microenvironments through higher-order spatial multi-omics integration

## 1. Requirements
+ Python == 3.8


## 2. Datasets
All datasets used in this study are publicly available and obtained from two previously published spatial multi-omics studies [[1]](https://www.nature.com/articles/s41587-023-01937-y) and [[2]](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202502028). Please find and download the data from:
+ [Fig.2] Mouse striatum (FMP-10): [sma/V11L12-109/V11L12-109_B1](https://data.mendeley.com/datasets/w7nw4km7xd/1)
+ [Fig.3] Mouse substantia nigra (FMP-10): [sma/V11T16-085/V11T16-085_B1](https://data.mendeley.com/datasets/w7nw4km7xd/1)
+ [Fig.4] Human lung cancer: [LC_091.h5ad](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZFCWC)
+ [Fig.5] Human breast cancer: [BC_515_Section_1.h5ad](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZFCWC)
+ [Fig.S6] Mouse striatum (DHB): [sma/V11L12-038/V11L12-038_A1](https://data.mendeley.com/datasets/w7nw4km7xd/1)
+ Cell-type annotations for the three mouse datasets were obtained from the original study in [[1]](https://www.nature.com/articles/s41587-023-01937-y) and can be downloaded from: [Zeisel_stsc_output.csv](https://github.com/Elsa-bingxue/SHINE/tree/main/Mouse%20striatum%20(FMP-10))
+ High-quality single-cell–derived cell-type annotations for human breast cancer were obtained from [[3](https://www.nature.com/articles/s41588-021-00911-1#citeas)] and [GSE176078/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078#:~:text=Series%20GSE176078%20%20%2011,profiling%20by%20high%20throughput%20sequencing)

## 3. Step-by-step Running

### 3.1 Data preparation and alignment

1. **Download the datasets** (see Section 2) and place them into the corresponding folders.

2. **Place the core scripts** in the project directory:
   - `Training.py`
   - `model.py`
   - `preprocess.py`

3. **Prepare aligned inputs and graph data**:

   - Open the Jupyter notebook used for data preparation (e.g., `create_data.ipynb`).
   - Execute all cells sequentially to perform cross-modal data registration and graph construction.

   This step generates the aligned inputs and graph-structured data required by SHINE.
   
### 3.2 Train SHINE to obtain embeddings

python Training.py

The learned SHINE embeddings will be saved automatically.

### 3.3 Run downstream analysis using saved embeddings
 - Open the Jupyter notebook in the Saved_Embedding_Analysis/ directory.

 - Execute all cells sequentially to perform downstream analysis based on the saved SHINE embeddings.

This step includes clustering, visualization, and embedding-based analyses.


# References
[1] Vicari, M. et al. Spatial multimodal analysis of transcriptomes and metabolomes in tissues. *Nature Biotechnology* 42, 1046–1050 (2024). https://doi.org/10.1038/s41587-023-01937-y.

[2] Godfrey, T. M. et al. Integrating ambient ionization mass spectrometry imaging and spatial transcriptomics on the same cancer tissues to identify rna–metabolite correlations. *Angewandte Chemie International Edition* 64, e202502028 (2025).  https://doi.org/10.1002/anie.202502028.

[3] Wu, S. Z. et al. A single-cell and spatially resolved atlas of human breast cancers. *Nature Genetics* 53, 1334–1347 (2021). https://doi.org/10.1038/s41588-021-00911-1
