python3 train_jaav2_lstm.py \
        --task_fold 'DISFA_combine_1_3' \
        --pretrain_prefix './models/weights/DISFA_combine_1_3'  \
        --train_path_prefix './data/list/DISFA_combine_1_3' \
        --test_path_prefix './data/list/DISFA_part2'  \
        --gpu_id 0