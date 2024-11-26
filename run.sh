#!/bin/bash

seeds=( {1..10..1} )
exp="exp_2"
device=3
# models=( MHGCN NeuroPath Mew GCN SAGE SGC GAT GCN_fuse_embed SAGE_fuse_embed SGC_fuse_embed GAT_fuse_embed )
models=( NeuroPath GAT GCN_fuse_embed SAGE_fuse_embed SGC_fuse_embed GAT_fuse_embed )
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        # CUDA_VISIBLE_DEVICES=$device python run_wandb.py --wandb normal --config ./configs/${model}_best.yaml --project_name ${exp} --seed $seed &
        echo $device
        device=$(( device + 1 ))
        if [ ${device} -eq 8 ]; then
            device=3
        fi
    done
    wait
done
# # wait 
# CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 0 &
# CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 1 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 2 &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 3 &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb normal --config ./configs/Mew_best.yaml --project_name exp_1 --seed 4 &
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