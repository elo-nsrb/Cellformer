## 1. Peak calling

Rscript 1-preprocessing/peakCalling.R --path_data [directory including arrow files] --output [output path] --metadata [csv file with cell annotations]


## 2. Create snATAC-seq peak Matrix

python 1-preprocessing/createPeakMatrix.py --path [path to data saved un 1.]


## 3. Create synthetic normalized data

python 1-preprocessing/createSyntheticDataset.py --path [path to data saved un 1.]
python 1-preprocessing/normalizePerCellCount.py --path [path to data saved un 1.]

## Annotate peaks and merge with ArchR annotations

Rscript 1-preprocessing/annotPeakChipSeeker.R --path_data ./data/

bash convertTSV2SAF.sh
python 1-preprocessing/addPeakHeader.py
python 1-preprocessing/mergeAnnotations.py

## Create bulk ATAC-seq peak matrix

bash 1-preprocessing/runFeatureCounts.sh



