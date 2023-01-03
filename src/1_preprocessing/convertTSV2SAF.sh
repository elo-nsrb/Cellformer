#!/bin/bash
MYPATH="./data/"
awk 'OFS="\t" {print $1"."$2"."$3,$1,$2,$3, "."}' $MYPATH/peaks.tsv > $MYPATH/peaks.saf
