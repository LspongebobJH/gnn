# Install packages
* pytorch==2.3.0
* dgl==2.2.0
* torch-geometric==2.6.1
* torch_scatter==2.1.2
* torchmetrics==1.2.1
* wandb==0.18.7
* scikit-learn==1.4.0

# How to run

* device: specify which cuda device that model is trained on
* config: specify hyperparameters of model training
* project: logging project name in wandb
* seed: random seed
* mode: if mode==normal, running a single experiment; if mode==sweep, running a hyperparameter tuning sweep.
  
```bash
CUDA_VISIBLE_DEVICES=${device} python run_wandb.py --wandb ${mode} --config ${config} --project_name ${project} --seed ${seed}

# e.g. Run a single experiment
CUDA_VISIBLE_DEVICES=0 python run_wandb.py --wandb normal --config ./configs/SAGE_best.yaml --project_name multiplex-gnn --seed 0

# e.g. Run 10 repeated experiments with the same hyperparameter setting but different seeds
run() {
    local model="$1"  # Capture the first argument
    seeds=( {0..9..1} )
    device=0
    config="./configs/${model}_best.yaml"
    project="multiplex-gnn"
    for seed in "${seeds[@]}"; do
        CUDA_VISIBLE_DEVICES=${device} python run_wandb.py --wandb normal --config ${config} --project_name ${project} --seed $seed
    done
}

run(GCN)

# e.g. Run a hyperparameter tuning sweep
CUDA_VISIBLE_DEVICES=0 python run_wandb.py --wandb sweep --config ./configs/GNN_wandb.yaml --project_name multiplex-tune --seed 0
```

### Notes
`./configs/GNN_wandb.yaml` is used to run wandb hyperparameter tuning. Hyperparameter candidates in this file should be specified according to each particular model hyperparameter specified in `./config/${model}_best.yaml` since tunable hyperparameters of models are different. For instance, if you want to tune hyperparameters of GCN, please refer to `./config/GCN_best.yaml`.

# How to reproduce experiments
All experiments can be reproduced by running following commands and results are shown in wandb with project name "multiplex-gnn". Note that the function `run()` is defined above.

```bash
# MewFuseGraph on all data (GNN)
run("MewFuseGraph_fuse_method_GAT_missLabel_labelPropFalse")

# MewFuseGraph on labeled data (GNN)
run("MewFuseGraph_fuse_method_SAGE")

# MewFuseGraph on labeled data (mean)
run("MewFuseGraph_fuse_method_mean")

# model=MHGCN, NeuroPath, Mew, GCN, SAGE, SGC, GAT, {GCN, SAGE, SGC, GAT}_fuse_embed,
run("${model}")
```

# Experiment results
All numbers are mean and std of 10 repeated experiments across seeds 0 ~ 9, which should be reproduced by commands above.

| Model Name                                        | Mean ± Std     |
|---------------------------------------------------|-----------------|
| MewFuseGraph on all data (GNN)                    | 16.24 ± 1.19    |
| MHGCN                                             | 16.26 ± 1.21    |
| MewFuseGraph on labeled data (GNN)                | 16.26 ± 1.19    |
| NeuroPath                                        | 16.28 ± 1.20    |
| MewFuseGraph on labeled data (mean)               | 16.29 ± 1.21    |
| Mew                                               | 16.31 ± 1.18    |
| SAGE_fuse_embed                                   | 16.41 ± 1.25    |
| SAGE                                              | 16.50 ± 1.37    |
| SGC_fuse_embed                                    | 16.58 ± 1.28    |
| SGC                                               | 16.78 ± 1.02    |
| GCN_fuse_embed                                    | 16.79 ± 1.15    |
| GCN                                               | 19.07 ± 0.96    |
| GAT                                               | 19.10 ± 4.43    |
| GAT_fuse_embed                                    | 21.13 ± 13.39   |

# File structure
There are some file and folders not listed below since they are unimportant or deprecated for experiments.
```
.
├── CIVAE.py: run model CIVAE, adapted from (https://github.com/kyg0910/CI-iVAE)
├── CIVAE_src: utils for CIVAE.py, adapted from (https://github.com/kyg0910/CI-iVAE)
├── DMGI.py: run model DMGI, adapted from (https://github.com/pcy1302/DMGI)
├── DMGI_src: utils for DMGI, adapted from (https://github.com/pcy1302/DMGI)
├── GNN.py: run a series of GNN models, including [GCN, SAGE, GAT, SGC, MHGCN, NeuroPath, Mew] and their variants.
├── MHGCN_src: utils for MHGCN, adapted from (https://github.com/NSSSJSS/MHGCN)
├── NeuroPath_src: utils for NeuroPath, adapted from (https://github.com/Chrisa142857/neuro_detour)
├── README.md
├── configs: config files
│   ├── *_best.yaml: best hyperparameter setting based on tuning sweep results
│   ├── *_wandb.yaml: custom config files used for wandb hyperparameter tuning
├── custom_src
│   ├── __init__.py
│   ├── fuse_models.py: implement multiplex variants of GNN using graph, embedding and prediction fusion, where different graph layers share the same learnable parameters.
│   ├── fuse_models_nosia.py: same as fuse_models.py, with the only difference that different graph layers have different learnable parameters.
│   ├── mew.py: Implementations of model Mew, adapted from (https://github.com/UNITES-Lab/Mew)
│   └── mew_fuse_graph.py: integrate graph and embedding fusion modules into Mew, adapted from mew.py
├── dataset: where original and processed data are stored
├── gen_valid_test.py: generate valid and test datasets
├── run_wandb.py: the main run file
├── requirements.txt
└── utils.py
```
