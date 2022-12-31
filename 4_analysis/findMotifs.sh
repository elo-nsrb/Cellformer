#!/bin/bash
MYPATH=../cellformer/deconvoluted/deconvoluted/bed_diff

background=$MYPATH/*all.bed
for f in $MYPATH/*0.01.bed
do
    filename=$(basename $f)
    echo $filename
    ~/bin/findMotifsGenome.pl $f hg38 $MYPATH/peak_homer/$filename -bg $background
done

