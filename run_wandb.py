import wandb
import yaml

project_name = 'graph_test'

def main():
    wandb.init(project=project_name)
    config = wandb.config
    if config.model_name in \
        ['MHGCN', 'NeuroPath', 'GCN', 'SAGE', 'SGC', 'GAT', 'Transformer']:
        from GNN import pipe
    elif config.model_name in ['CIVAE']:
        from CIVAE import pipe
    elif config.model_name in ['DMGI']:
        from DMGI import pipe
    pipe(config)

if __name__ == '__main__':


    with open('./configs/GNN.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=configs, project=project_name)
    wandb.agent(sweep_id, function=main)