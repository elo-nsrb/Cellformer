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
Rscript src/1-preprocessing/peakCalling.R \
 --path_data [directory including arrow files] \
 --output [path to save output files] \
 --metadata [csv file with cell annotations]

python src/1-preprocessing/createPeakMatrix.py \
 --path [path with previously saved files]
 ```

### 2. Create synthetic normalized data
Synthetic dataset can be created from snATAC-seq peak matrix in [AnnData format](https://anndata.readthedocs.io/en/latest/) (see example in [data](https://github.com/elo-nsrb/Cellformer/tree/main/data)):

```
python src/1-preprocessing/createSyntheticDataset.py \
 --path [directory containing the peak matrix] \
 --filename [peak matrix file]

python 1-preprocessing/normalizePerCellCount.py 
 --path [directory containing the peak matrix] 
 --filename [peak matrix file]
 ```

### 3. Train Cellformer and deconvolute bulk
