#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run_wandb.py --wandb normal --config ./configs/GAT_fuse_embed_best.yaml --project_name exp_1 --seed 0 &
CUDA_VISIBLE_DEVICES=4 python run_wandb.py --wandb normal --config ./configs/GAT_fuse_embed_best.yaml --project_name exp_1 --seed 1 &
CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/GAT_fuse_embed_best.yaml --project_name exp_1 --seed 2 &
CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/GAT_fuse_embed_best.yaml --project_name exp_1 --seed 3 &
CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/GAT_fuse_embed_best.yaml --project_name exp_1 --seed 4 &