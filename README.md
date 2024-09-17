# Cellformer
An implementation of Cellformer from our publication: Berson et al. *"Whole genome deconvolution unveils Alzheimer’s resilient epigenetic signature"*

## Installation from source
The lastest source version of Cellformer can be accessed by running the following command:

```
git clone https://github.com/elo-nsrb/Cellformer.git
cd Cellformer
```

## Requirements

* Python 3
* PyTorch (2.0.0)
* ArchR (R version 4.2.2)
* Scikit-learn (1.2.2)
* Asteroid (0.5.2)
* Pytorch-lightning (2.0.4)
* comet_ml (3.32.8)

In order to install package dependencies, you will need [Anaconda](https://anaconda.org/). After installing Anaconda, please run the following command to create two conda environnements with R and Pytorch dependencies:

`.\setup.sh`

## Usage
### 1. Peak calling and peak matrix creation

To create a peak matrix from single-cell ATAC-seq fragment files, please use the following commands:

```
conda activate R_env
./createPeakMatrix.sh --path_data ./data/ --input_dir ./scDATA/ --metadata ./data/scATAC_cell_annotations.csv
```


```
usage: createDataset -p | --path_data PATH_DATA
                     -i | --input_dir INPUT_DIR
                     -m | --metadata METADATA
                     [ -h | --help]
positional arguments:
-p, --path_data         Path to save the peak Matrix
-i, --input_dir         Directory with single cell arrow files
-m, --metadata          Metadata with cell annotations

optional arguments:
-h, --help              show help message and exit
```

### 2. Synthetic dataset generation
Synthetic dataset can be created from snATAC-seq peak matrix in [AnnData format](https://anndata.readthedocs.io/en/latest/) with `celltype` and `Sample_num` columns in `obs`(see example in [data](https://github.com/elo-nsrb/Cellformer/tree/main/data)):

```
./createDataset.sh -p ../data/ -n 500
```

```
usage: createDataset -p | --path_data PATH_DATA
                     -n | --nbSamplesPerCase NBSAMPLESPERCASE
                     [ -f | --matrixfilename MATRIXFILEMANE]
                     [ -h | --help]
positional arguments:
-p, --path_data         Path to directory with the peak matrix
-n, --nbCellsPerCase    Number of synthetic samples per individual

optional arguments:
-h, --help              Show help message and exit
-f, --matrixfilename    Name of peak matrix file, default=adata_peak_matrix.h5
```

### 3. Pretrained model inference and bulk deconvolution
We provided the pretrained model used in the manuscript in [cellformer](https://github.com/elo-nsrb/Cellformer/tree/main/cellformer). The pretrained model can be used to deconvolute bulk peak matrix by running:

```
conda activate pytorch_env
./deconvolution --model_path cellformer/ --peak_matrix ./data/CTRL_CAUD_AD.peak_countMatrix.txt
```

```
Usage: deconvolution  -p | --model_path MODEL_PATH
                      -m | --peak_matrix PEAK_MATRIX
                      [ -h | --help  ]"
positional arguments:
-p, --model_path        Path to model directory with train.yml
-m, --peak_matrix       Peak matrix to deconvolute

optional arguments:
-h, --help              Show help message and exit
```
You can find an example of the expected peak matrix format `CTRL_CAUD_AD.peak_countMatrix.txt` in [data](https://github.com/elo-nsrb/Cellformer/tree/main/data).

### 4. Model training

Cellformer can be trained from scratch using a synthetic dataset and configuration file `train.yml` (see an example in [cellformer](https://github.com/elo-nsrb/Cellformer/tree/main/cellformer)) by running:
```
conda activate pytorch_env
./trainModel.sh --model_path cellformer/
```

```
usage: trainModel -p | --model_path MODEL_PATH
                  [ -h | --help]
positional arguments:
-p, --model_path        Path to model directory with train.yml

optional arguments:
-h, --help              Show help message and exit
```
Please modify the path to the data folder in `train.yml`.

### 5. Model testing using trained model
Model testing can done using bulk or pseudobulk data by running:

```
./validation.sh --model_path cellformer/ 
```

```
Usage: validationModel  -p | --model_path MODEL_PATH
                        [ -h | --help  ]
positional arguments:
-p, --model_path        Path to model directory with train.yml

optional arguments:
-h, --help              Show help message and exit
```
Please note that in the new version, the mask is computed on the cross validation performance matrix. We recommend to remove cell type specific peaks with a spearman correlation <0.2 across samples.
## Licence
This project is covered under the [GNU General Public License v3.0](https://github.com/elo-nsrb/Cellformer/blob/main/LICENSE)
