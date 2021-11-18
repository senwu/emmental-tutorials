#!/bin/bash
# This script is for running our text classification approach.
# Usage: bash run_text_classification.sh ${TASK} ${DATA} ${MODEL} ${LOGPATH} ${SEED} ${GPU_ID} ${EMB}
#   - TASK: task name list delimited by ",". Defaults to all.
#   - DATA: data directory. Defaults to "data".
#   - MODEL: model. Defaults to "cnn".
#   - LOGPATH: log directory. Defaults to "logs".
#   - SEED: random seed. Defaults to 111.
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.
#   - EMB: Embedding file path. Defaults to glove.6B.300d.txt.

TASK=${1:-mr,sst,subj,cr,mpqa,trec}
DATA=${2:-data}
MODEL=${3:-lstm}
LOGPATH=${4:-logs}
SEED=${5:-1}
GPU=${6:-0}
EMB=${7:-glove.6B.50d.txt}
DIM=${8:-50}
NEPOCHS=${9-20}
LR=${10-1e-3}
L2=${11-0}
NFILETERS=${12-100}

python run.py \
  --task ${TASK} \
  --seed ${SEED} \
  --data_dir ${DATA} \
  --log_path ${LOGPATH} \
  --device ${GPU} \
  --model ${MODEL} \
  --n_epochs ${NEPOCHS} \
  --train_split train \
  --valid_split valid \
  --optimizer adam \
  --lr ${LR} \
  --l2 ${L2} \
  --warmup_percentage 0.0 \
  --dataparallel 0 \
  --counter_unit epoch \
  --evaluation_freq 1 \
  --checkpoint_freq 1 \
  --checkpointing 0 \
  --checkpoint_metric model/all/valid/macro_average:max \
  --checkpoint_task_metrics mr/mr/valid/accuracy:max,sst/sst/valid/accuracy:max,subj/subj/valid/accuracy:max,cr/cr/valid/accuracy:max,mpqa/mpqa/valid/accuracy:max,trec/trec/valid/accuracy:max \
  --batch_size 32 \
  --min_data_len 5 \
  --embedding ${EMB} \
  --dim ${DIM} \
  --fix_emb 0 \
  --dataparallel 0 \
  --n_filters ${NFILETERS} \
  --clear_intermediate_checkpoints 1 \
  --clear_all_checkpoints 0
