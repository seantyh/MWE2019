# !/bin/bash

PWD=$(pwd)
source /home/seantyh/langon/venv/bin/activate
SCRIPT_DIR=`dirname $0`
cd "$SCRIPT_DIR/../"
python main.py --task variation --corpus apple --sample-ratio 0.01
mkdir -p data/variations
docker exec mwe_mongo bash \
    -c 'exec mongoexport -d mwe -c apple --pretty' \
    > data/variations/apple_vars.json
cd $PWD




