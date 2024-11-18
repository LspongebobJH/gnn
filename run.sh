#!/bin/bash

# seeds=( {1..5..1} )

# for seed in "${seeds[@]}"; do
# CUDA_VISIBLE_DEVICES=1 python run_wandb.py --wandb normal --config ./configs/NeuroPath_best.yaml --project_name exp_1 --seed 0 &
# CUDA_VISIBLE_DEVICES=1 python run_wandb.py --wandb normal --config ./configs/NeuroPath_best.yaml --project_name exp_1 --seed 1 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/NeuroPath_best.yaml --project_name exp_1 --seed 2 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/NeuroPath_best.yaml --project_name exp_1 --seed 3 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/NeuroPath_best.yaml --project_name exp_1 --seed 4 &

# wait 
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/SGC_best.yaml --project_name exp_1 --seed 0 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/SGC_best.yaml --project_name exp_1 --seed 1 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/SGC_best.yaml --project_name exp_1 --seed 2 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/SGC_best.yaml --project_name exp_1 --seed 3 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/SGC_best.yaml --project_name exp_1 --seed 4 &


CUDA_VISIBLE_DEVICES=2 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb.yaml &
CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_1.yaml &