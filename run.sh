#!/bin/bash

seeds=( {0..9..1} )
exp="exp_4"
device=4
# models=( MHGCN NeuroPath Mew GCN SAGE SGC GAT GCN_fuse_embed SAGE_fuse_embed SGC_fuse_embed GAT_fuse_embed )
# models=( MHGCN NeuroPath Mew GCN SAGE SGC GCN_fuse_embed SAGE_fuse_embed SGC_fuse_embed )
# models=( MewFuseGraph_fuse_method_GCN_selfLoop MewFuseGraph_fuse_method_SAGE_selfLoop)
models=( MewFuseGraph_fuse_method_GAT_selfLoop)
# models=(Mew_custom)
cnt=0
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        CUDA_VISIBLE_DEVICES=$device python run_wandb.py --wandb normal --config ./configs/${model}_best.yaml --project_name ${exp} --seed $seed &

        cnt=$(( cnt + 1 ))
        _cnt=$(( cnt % 5 ))
        if [ ${_cnt} -eq 0 ]; then
            wait
        fi
        # device=$(( device + 1 ))
        # if [ ${device} -eq 8 ]; then
        #     device=
        #     # wait
        # fi
    done
    wait
done
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

# CUDA_VISIBLE_DEVICES=5 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb.yaml --project_name MewFuseGraph &
# CUDA_VISIBLE_DEVICES=2 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_1.yaml --project_name graph_test &
# CUDA_VISIBLE_DEVICES=6 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_1.yaml --project_name graph_test &
# CUDA_VISIBLE_DEVICES=7 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_2.yaml --project_name graph_test &