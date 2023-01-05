# Cellformer
An implementation of Cellformer from our publication: Berson et al. *"Whole genome deconvolution unveils Alzheimer’s resilient epigenetic signature"*

## Installation from source
The lastest source version of Cellformer can be accessed by running the following command:

```
git clone https://github.com/elo-nsrb/Cellformer.git
```


## Usage
### 1. Peak calling and peak matrix creation

To create a peak matrix compatible with Cellformer from single-cell ATAC-seq fragment files, please use the following commands:

```
./createPeakMatrix.sh -p ./data/ -i ./scDATA/ -m ./data/scATAC_cell_annotations.csv

usage: createDataset -p | --path_data PATH_DATA
                     -i | --input_dir INPUT_DIR
                     -m | --metadata METADATA
                     [ -h | --help]
positional arguments:
-p, --path_data         path to save the peak Matrix
-i, --input_dir         directory including the input arrow files
-m, --metadata          metadata with cell annotations

optional arguments:
-h, --help              show help message and exit
```

### 2. Create synthetic normalized data
Synthetic dataset can be created from snATAC-seq peak matrix in [AnnData format](https://anndata.readthedocs.io/en/latest/) (see example in [data](https://github.com/elo-nsrb/Cellformer/tree/main/data)):

```
./createDataset.sh -p ../data/ -n 500

usage: createDataset -p | --path_data PATH_DATA
                     -n | --nbSamplesPerCase NBSAMPLESPERCASE
                     [ -f | --matrixfilename MATRIXFILEMANE]
                     [ -h | --help]
positional arguments:
-p, --path_data         path to directory including peak Matrix
-n, --nbCellsPerCase    Number of synthetic samples per individual

optional arguments:
-h, --help              show help message and exit
-f, --matrixfilename    name of peak matrix file, default=adata_peak_matrix.h5
```

### 3. Train Cellformer and deconvolute bulk
