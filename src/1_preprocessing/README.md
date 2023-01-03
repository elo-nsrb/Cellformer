#Additional tools

### Annotate peaks and merge with ArchR annotations

`Rscript 1-preprocessing/annotPeakChipSeeker.R --path_data ./data/

bash convertTSV2SAF.sh
python 1-preprocessing/addPeakHeader.py
python 1-preprocessing/mergeAnnotations.py`

## Create bulk ATAC-seq peak matrix

`bash 1-preprocessing/runFeatureCounts.sh`



