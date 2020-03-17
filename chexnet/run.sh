DATAPATH=${1}
IMAGEPATH=${2}
TASK=${3:-Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia}
LOG=${4:-logs}
SEED=${5:-0}
GPU=${6:-0}
LR=${7:-0.001}

python run.py \
    --task ${TASK} \
    --data_path ${DATAPATH} \
    --image_path ${IMAGEPATH} \
    --log_path ${LOG} \
    --device ${GPU} \
    --seed ${SEED} \
    --model densenet121 \
    --n_epochs 100 \
    --optimizer sgd \
    --train_split train \
    --valid_split val test \
    --optimizer sgd \
    --lr ${LR} \
    --l2 0.0001 \
    --sgd_momentum 0.9 \
    --lr_scheduler plateau \
    --lr_scheduler_step_unit epoch \
    --reset_state 1 \
    --plateau_lr_scheduler_metric model/all/val/loss \
    --plateau_lr_scheduler_mode min \
    --plateau_lr_scheduler_patience 0 \
    --plateau_lr_scheduler_eps 0 \
    --dataparallel 0 \
    --evaluation_freq 1 \
    --checkpointing 1 \
    --checkpoint_metric model/all/val/loss:min \
    --clear_intermediate_checkpoints 1 \
    --clear_all_checkpoints 1 \
    --batch_size 16
