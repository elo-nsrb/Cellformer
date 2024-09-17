#!/bin/bash

help()
{
    echo "Usage: trainModel  -p | --model_path MODEL_PATH
                              [ -h | --help  ]"
    exit 2
}
SHORT=p:,h
LONG=model_path:,help
OPTS=$(getopt -a -n trainModel --option $SHORT --longoptions $LONG -- "$@")


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

python src/2_deconvolution/cvTrain.py --model_path $model_path --model SepFormerTasNet

echo "Model trained!"

