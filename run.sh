#!/bin/bash

# seeds=( {1..5..1} )
# models=( GAT )
# for model in "${models[@]}"; do
#     CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/${model}_fuse_graph_best.yaml --project_name exp_1 --seed 0 &
#     CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/${model}_fuse_graph_best.yaml --project_name exp_1 --seed 1 &
#     CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/${model}_fuse_graph_best.yaml --project_name exp_1 --seed 2 &
#     wait
#     CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/${model}_fuse_graph_best.yaml --project_name exp_1 --seed 3 &
#     CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/${model}_fuse_graph_best.yaml --project_name exp_1 --seed 4 &
#     wait
# done
# # wait 
CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 0 &
CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 1 &
CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 2 &
CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 3 &
CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 4 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/SAGE_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 2 &

# wait

# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/SGC_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 3 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/SGC_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 4 &

# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/SAGE_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 3 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/SAGE_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 4 &

# wait

# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/GCN_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 0 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/GCN_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 1 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/GCN_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 2 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/GCN_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 3 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/GCN_fuse_embed_nosia_best.yaml --project_name exp_1 --seed 4 &

# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb.yaml --project_name graph_test &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_1.yaml --project_name graph_test &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_2.yaml --project_name graph_test &