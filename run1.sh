#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb_1.yaml &