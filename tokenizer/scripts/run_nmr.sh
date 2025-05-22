# #!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

set -x #关闭调试模式，脚本将不再打印每一行的执行过程。
torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    main.py \
    --max_length=2048 \
    --output_dir=weight/weight_01 \
    --gpus=${NUM_GPUS_PER_NODE} \
    --num_train_epochs=200 \
    --num_workers=${NUM_GPUS_PER_NODE} \
    --batch_size=1 \
    --learning_rate=0.00005 \
    --weight_decay=0.0001 \
    --num_warmup_epochs=1 \
    --use_molecular_formula_prob=0.5 \
    --use_1H_NMR_prob=0.8 \
    --use_13C_NMR_prob=1.0 \
    --use_COSY_prob=0.5 \
    --use_HMBC_prob=0.5 \
    --use_HH_prob=0.5 \
    --use_smiles_prob=1.0 \
    --config_json_path=configs/bart.json \
    --seed=42 \
    --do_train \
    --do_test \
    --input_name=1H_NMR,13C_NMR,COSY,HMBC,HH,molecular_formula \
    --output_name=smiles \
    --mode=forward \
    --num_beams=10 \
    --tokenizer_path=tokenizer \
    --train_folder=dataset/20w_20240904/train_20240701_part1.json \
    --validation_folder=dataset/20w_20240904/val_20240701.json \
    --test_folder=dataset/20w_20240904/test_20240701.json \
    --model_weight=weight/weight_02/epoch_89_loss_0.066344.pth \
    