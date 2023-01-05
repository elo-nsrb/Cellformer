#!/bin/bash

help()
{
    echo "Usage: validationModel  -mp | --model_path MODEL_PATH
                                  -m | --peak_matrix PEAK_MATRIX
                                  -g | --groundtruth GROUNDTRUTH
                                  [ -h | --help  ]"
    exit 2
}
SHORT=p:,m:,g:,h
LONG=model_path:,peak_matrix:,groundtruth:,help
OPTS=$(getopt -a -n validationModel --option $SHORT --longoptions $LONG -- "$@")


VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
      help
fi

eval set -- "$OPTS"

while :
do
    case "$1" in
        -p | --model_path )
        model_path="$2"
        shift 2
        ;;
        -m | --peak_matrix )
        peak_matrix="$2"
        shift 2
        ;;
        -g | --groundtruth )
        groundtruth="$2"
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

python src/2_deconvolution/inference.py --model_path $model_path --model SepFormerTasNet --peak_count_matrix $peak_matrix --groundtruth $groundtruth --type pseudobulk --mask

echo "Validation done!"
