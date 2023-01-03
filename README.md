# Cellformer
An implementation of Cellformer from our publication: Berson et al. *"Whole genome deconvolution unveils Alzheimer’s resilient epigenetic signature"*

## Installation

##
### 1. Peak calling and peak matrix creation

To create a peak matrix compatible with Cellformer from fragment files, please use the following commands:

` Rscript src/1-preprocessing/peakCalling.R \
 --path_data [directory including arrow files] \
 --output [path to save output files] \
 --metadata [csv file with cell annotations] \

python src/1-preprocessing/createPeakMatrix.py \
 --path [path with previously saved files]`

### 2. Create synthetic normalized data
Synthetic dataset can be created from snATAC-seq peak matrix in annData format (see examples in [data](https://github.com/elo-nsrb/Cellformer/tree/main/data)):

`python 1-preprocessing/createSyntheticDataset.py \
 --path [path to data saved un 1. or location of the annData file] \
 --filename [name of the annData file]

python 1-preprocessing/normalizePerCellCount.py 
 --path [path to data saved un 1. or location of the annData file] 
 --filename [name of the annData file]`

### 3. Train Cellformer and deconvolute bulk
