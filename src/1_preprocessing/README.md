# Additional tools

### Annotate peaks and merge with ArchR annotations

We annotated peak.tsv from ArchR using the followind commands:
`python addPeakHeader.py --path_data ./data/
Rscript annotPeakChipSeeker.R --path_data ./data/
python mergeAnnotations.py --path_data ./data/`

Please set `path_data` as the path to the folder with peak matrix and synthetic data.

### Create bulk ATAC-seq peak matrix

We created bulk peak matrix using `FeatureCounts`:
`bash 1-preprocessing/runFeatureCounts.sh`



