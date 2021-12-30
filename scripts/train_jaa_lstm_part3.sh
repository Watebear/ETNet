python3 train_jaav2_lstm.py \
        --task_fold 'DISFA_combine_1_2' \
        --pretrain_prefix './models/weights/DISFA_combine_1_2'  \
        --train_path_prefix './data/list/DISFA_combine_1_2' \
        --test_path_prefix './data/list/DISFA_part3'  \
        --gpu_id 0