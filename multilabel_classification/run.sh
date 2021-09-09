DATAPATH=${1:-data}
MODEL=${2:-distilbert-base-uncased}
LOG=${3:-logs}
SEED=${4:-0}
GPU=${5:-1}
LR=${6:-0.00003}
MAX_SAMPLE=${7:-1000}

python run.py \
    --data_path ${DATAPATH} \
    --log_path ${LOG} \
    --device ${GPU} \
    --seed ${SEED} \
    --model ${MODEL} \
    --n_epochs 5 \
    --optimizer adamw \
    --train_split train \
    --valid_split test \
    --lr ${LR} \
    --lr_scheduler linear \
    --warmup_percentage 0.1 \
    --dataparallel 0 \
    --evaluation_freq 0.2 \
    --checkpointing 1 \
    --online_eval 1 \
    --checkpoint_metric toxic/toxic/test/f1:max \
    --clear_intermediate_checkpoints 1 \
    --batch_size 128
    # --max_data_samples ${MAX_SAMPLE}
