python multirc.py --task MultiRC --seed 1 --log_path logs --n_epochs 3 --train_split train --valid_split val --lr 1e-5 --warmup_percentage 0.0 --counter_unit epoch --evaluation_freq 0.1 --checkpoint_freq 1 --checkpoint_metric model/train/all/loss:min --checkpoint_task_metrics MultiRC/SuperGLUE/val/em_f1:max --checkpoint_runway 0.5 --checkpoint_clear True --data_dir data --bert_model bert-large-cased --batch_size 4 --max_sequence_length 128 --max_data_samples 1 --device -1 --dataparallel 0
