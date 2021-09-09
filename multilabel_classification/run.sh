TASK_NAME=${1:-toxic}
TRAIN_DATA_PATH=${2:-data/train.csv}
VAL_DATA_PATH=${3:-none}
TEST_DATA_PATH=${4:-data/test.csv}
INPUT_FIELD=${5:-comment_text}
LABEL_FIELDS=${6:-toxic,severe_toxic,obscene,threat,insult,identity_hate}
MODEL=${7:-distilbert-base-uncased}
LOG=${8:-logs}
SEED=${9:-0}
LR=${10:-0.00003}
EPOCH=${11:-5}
MAX_DATA_SAMPLE=${12:-none}
MAX_SEQ_LENGTH=${13:-128}

python run.py \
    --task_name ${TASK_NAME} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --val_data_path ${VAL_DATA_PATH} \
    --test_data_path ${TEST_DATA_PATH} \
    --input_field ${INPUT_FIELD} \
    --label_fields ${LABEL_FIELDS} \
    --model ${MODEL} \
    --log_path ${LOG} \
    --seed ${SEED} \
    --n_epochs ${EPOCH} \
    --optimizer adamw \
    --train_split train \
    --valid_split val test \
    --lr ${LR} \
    --lr_scheduler linear \
    --warmup_percentage 0.1 \
    --dataparallel 0 \
    --evaluation_freq 0.2 \
    --checkpointing 1 \
    --online_eval 1 \
    --checkpoint_metric ${TASK_NAME}/${TASK_NAME}/test/f1:max \
    --clear_intermediate_checkpoints 1 \
    --batch_size 128 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_data_samples ${MAX_DATA_SAMPLE}
