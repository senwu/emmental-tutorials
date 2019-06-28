#!/bin/bash
# This script is for running our SuperGLUE approach.
# Usage: bash run_superglue.sh ${TASK} ${SUPERGLUEDATA} ${SEED} ${GPU_ID}
#   - TASK: one of {"cb", "copa", "multirc", "rte", "wic", "wsc", "swag"}
#       Note: swag is an external task used for copa pretraining
#   - SUPERGLUEDATA: SuperGLUE data directory. Defaults to "data".
#   - LOGPATH: log directory. Defaults to "logs".
#   - SEED: random seed. Defaults to 111.
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.

TASK=${1}
SUPERGLUEDATA=${2:-data}
LOGPATH=${3:-logs}
SEED=${4:-111}
GPU=${5:-0}

# Check TASK name
case ${TASK} in
    cb|copa|multirc|rte|wic|wsc)
        echo "TASK: ${TASK}"
        echo "DATA: ${SUPERGLUEDATA}"
        echo "LOGPATH: ${LOGPATH}"
        echo "SEED: ${SEED}"
        echo "GPU: ${GPU}" ;;
    *)
        echo "Unrecognized task ${TASK}..."
        exit 1 ;;
esac

if [ ${TASK} == "cb" ]; then
    python run.py \
        --task CB \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 10 \
        --train_split train \
        --valid_split val \
        --optimizer adam \
        --lr 1e-5 \
        --grad_clip 5.0 \
        --warmup_percentage 0.0 \
        --counter_unit epochs \
        --evaluation_freq 0.1 \
        --checkpointing 1 \
	--checkpoint_metric COPA/SuperGLUE/val/accuracy:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --bert_model bert-large-cased \
        --batch_size 4 \
        --max_sequence_length 256 \
        --dataparallel 0
elif [ ${TASK} == "copa" ]; then
    python run.py \
        --task COPA \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 10 \
        --train_split train \
        --valid_split val \
        --optimizer adam \
        --amsgrad 1 \
        --lr 1e-5 \
        --grad_clip 5.0 \
        --warmup_percentage 0.0 \
        --counter_unit epoch \
        --evaluation_freq 1 \
        --checkpoint_freq 1 \
	--checkpointing 1 \
        --checkpoint_metric COPA/SuperGLUE/val/accuracy:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --bert_model bert-large-cased \
        --batch_size 4 \
        --max_sequence_length 40 \
        --dataparallel 0
elif [ ${TASK} == "multirc" ]; then
    python run.py \
        --task MultiRC \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 10 \
        --train_split train \
        --valid_split val \
        --optimizer adam \
        --amsgrad 1 \
        --lr 1e-5 \
        --grad_clip 5.0 \
        --warmup_percentage 0.0 \
        --counter_unit batch \
        --evaluation_freq 1000 \
        --checkpoint_freq 1 \
	--checkpointing 1 \
        --checkpoint_metric MultiRC/SuperGLUE/val/em_f1:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --bert_model bert-large-cased \
        --batch_size 4 \
        --max_sequence_length 256 \
        --dataparallel 0
elif [ ${TASK} == "rte" ]; then
    python run.py \
        --task RTE \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 50 \
        --train_split train \
        --valid_split val \
	--optimizer adamax \
        --lr 2e-5 \
        --grad_clip 1.0 \
        --warmup_percentage 0.1 \
        --counter_unit epoch \
        --evaluation_freq 0.25 \
	--checkpoint_freq 1 \
	--checkpointing 1 \
        --checkpoint_metric RTE/SuperGLUE/val/accuracy:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --checkpoint_runway 1.0 \
        --bert_model bert-large-cased \
        --batch_size 8 \
        --max_sequence_length 256 \
        --slices 1 \
        --general_slices 1 \
        --dataparallel 0
elif [ ${TASK} == "wic" ]; then
    python run.py \
        --task WiC \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 20 \
        --train_split train \
	--valid_split val \
	--optimizer adam \
        --lr 1e-5 \
        --grad_clip 5.0 \
        --warmup_percentage 0.0 \
        --counter_unit epoch \
        --evaluation_freq 0.1 \
        --checkpoint_freq 1 \
	--checkpointing 1 \
        --checkpoint_metric WiC/SuperGLUE/val/accuracy:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --bert_model bert-large-cased \
        --batch_size 4 \
        --max_sequence_length 256 \
        --dataparallel 0
elif [ ${TASK} == "wsc" ]; then
    python run.py \
        --task WSC \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 20 \
        --train_split train \
        --valid_split val \
        --optimizer adam \
        --lr 1e-5 \
        --grad_clip 5.0 \
        --eps 1e-8 \
        --warmup_percentage 0.0 \
        --counter_unit epoch \
        --evaluation_freq 1 \
        --checkpoint_freq 1 \
	--checkpointing 1 \
        --checkpoint_metric WSC/SuperGLUE/val/accuracy:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --bert_model bert-large-cased \
        --batch_size 4 \
        --max_sequence_length 256 \
        --dataparallel 0
elif [ ${TASK} == "swag" ]; then
    python run.py \
        --task SWAG \
        --data_dir ${SUPERGLUEDATA} \
        --log_path ${LOGPATH} \
        --seed ${SEED} \
        --device ${GPU} \
        --n_epochs 5 \
        --train_split train \
        --valid_split val \
        --optimizer adamax \
        --lr 1e-5 \
        --eps 1e-8 \
        --grad_clip 5.0 \
        --warmup_percentage 0.0 \
        --counter_unit epoch \
        --evaluation_freq 0.1 \
        --checkpoint_freq 1 \
        --checkpointing 1 \
        --checkpoint_metric SWAG/SuperGLUE/val/accuracy:max \
        --checkpoint_task_metrics model/train/all/loss:min \
        --bert_model bert-large-cased \
        --batch_size 3 \
        --max_sequence_length 256 \
        --dataparallel 0 \
        --last_hidden_dropout_prob 0.1
fi
