#!/bin/bash

help()
{
    echo "Usage: createPeakMatrix [ -i | --input_dir INPUT_DIR]
                           [ -p | --path_data PATH_DATA ]
                            [ -m | --metadata METADATA]
                          [ -h | --help  ]"
    exit 2
}
SHORT=i:,p:,m:,h
LONG=input:,path_data:,metadata:,help
OPTS=$(getopt -a -n createPeakMatrix --option $SHORT --longoptions $LONG -- "$@")


VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
      help
fi

eval set -- "$OPTS"

while :
do
    case "$1" in
        -p | --path_data )
        path_data="$2"
        shift 2
        ;;
        -i | --input )
        input="$2"
        shift 2
        ;;
        -m | --metadata )
        metadata="$2"
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

Rscript src/1-preprocessing/peakCalling.R --path_data $input --output $path_data --metadata $metadata

python src/1-preprocessing/createPeakMatrix.py --path $path_data
