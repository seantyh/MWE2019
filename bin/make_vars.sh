#!/bin/bash

PWD=$(pwd)
source /home/seantyh/langon/venv/bin/activate
SCRIPT_DIR=`dirname $0`
cd "$SCRIPT_DIR/../"
python main.py --task variations --corpus apple
mkdir -p data/variations
docker exec mwe_mongo bash \
    -c 'exec mongoexport -d mwe -c apple --pretty' \
    > data/variations/apple_variations.json
cd $PWD




