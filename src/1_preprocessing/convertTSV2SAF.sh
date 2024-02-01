#!/bin/bash
MYPATH="/home/eloiseb/data/ATAC-seq_2024/"
awk 'OFS="\t" {print $1"."$2"."$3,$1,$2,$3, "."}' $MYPATH/peaks.tsv > $MYPATH/peaks.saf
