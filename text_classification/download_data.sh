#!/bin/bash
# This script is for downloading text classification datasets used in this tutorial.
# Usage: bash download_data.sh ${DATA}
#   - DATA: data directory. Defaults to "data".

DATA=${1:-data}

mkdir -p ${DATA}

echo "Download text classification datasets into ${DATA}..."

git clone https://github.com/harvardnlp/sent-conv-torch.git
cp -r sent-conv-torch/data/* ${DATA}
rm -rf sent-conv-torch

