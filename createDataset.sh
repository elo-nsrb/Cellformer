#!/bin/bash

help()
{
    echo "Usage: createDataset [ -p | --path ]
               [ -n | --nbSamplesPerCase ]
                [ -f | --matrixfilename MATRIXFILEMANE]
              [ -h | --help  ]"
    exit 2
}
SHORT=p:,n:,f:,h
LONG=path:,nbSamplesPerCase:,matrixfilename:,help
OPTS=$(getopt -a -n createDataset --option $SHORT --longoptions $LONG -- "$@")


VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
      help
fi

eval set -- "$OPTS"

while :
do
    case "$1" in
        -p | --path )
        pathData="$2"
        shift 2
        ;;
        -n | --nbSamplesPerCase )
        nbSamplesPerCase="$2"
        shift 2
        ;;
        -f | --matrixfilename )
        matrixfilename="$2"
        shift 2
        ;;
        -h | --help)
        help
        ;;
        --)
        shift;
        break
        ;;
        *)
        echo "Unexpected option: $1"
        help
        ;;
    esac
done
if [-z "$matrixfilename"]
then
    matrixfilename="adata_peak_matrix.h5"
fi
python 1_preprocessing/createSyntheticDataset.py --path $pathData --nb_cells_per_case $nbSamplesPerCase --nb_cores 10 --filename $matrixfilename

echo "Sample normalization ..."

python 1_preprocessing/normalizePerCellCount.py --path $pathData --filename $matrixfilename

echo "clean directory ..."
rm $pathData/pseudobulks_nb_cell*
rm $pathData/pseudobulks_celltype_specific_subject*
rm $pathData/pseudobulks_labels_data_subject*
rm $pathData/pseudobulks_data_subject*

echo "Done!"
