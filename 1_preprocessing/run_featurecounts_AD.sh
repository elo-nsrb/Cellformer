#!/bin/bash
PROJECT_ROOT=/home/eloiseb/stanford_drive/data/ATAC-seq
MYPATH=./data/
OUT_DIR=$MYPATH/count_from_sc_my_peaks_AD
ANNOT=$MYPATH/peaks.saf


mkdir $OUT_DIR
mkdir $OUT_DIR/AD
for type in $PROJECT_ROOT/AD/*; do
    for region in $type/*; do
        
	    type=$(basename $type)
        echo $type
	    region=$(basename $region)
        echo $region
	    OUTPUT_DIR=$OUT_DIR/AD/$type/
        mkdir $OUTPUT_DIR
	    echo $OUT_DIR
        echo $file
        echo $OUTPUT_DIR${type}_${region}_AD.peak_countMatrix.txt
        echo $PROJECT_ROOT/ATAC_seq/${type}_${region}_AD_*.bam
        echo $ANNOT
        ~/subread/bin/featureCounts -a $ANNOT -F SAF -p -o $OUTPUT_DIR${type}_${region}_PD.peak_countMatrix.txt $PROJECT_ROOT/ATAC_seq/${type}_${region}_PD_*.bam
        rm $OUTPUT_DIR*.tmp
    done
done
wait

echo "all done"
