awk 'OFS="\t" {print $2"."$3+1"."$4, $2, $3+1, $4, "."}' peaks.tsv > peaks.saf
