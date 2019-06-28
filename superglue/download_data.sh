#!/bin/bash
# This script is for downloading SuperGLUE and other datasets used in this tutorial.
# Usage: bash download_data.sh ${DATA}
#   - DATA: data directory. Defaults to "data".

DATA=${1:-data}

echo "Download datasets into ${DATA}..."

echo "Download SuperGLUE data..."

python download_superglue_data.py --data_dir ${DATA} --tasks all

echo "Download SWAG data..."

git clone https://github.com/rowanz/swagaf.git
cp -r swagaf/data/ ${DATA}/SWAG
rm -rf swagaf

