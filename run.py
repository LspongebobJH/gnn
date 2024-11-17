import wandb

if __name__ == '__main__':
    if wandb.run is not None:
        config = wandb.config
        wandb.init(
            config
        )
        if config.model_name in \
            ['MHGCN', 'NeuroPath', 'GCN', 'SAGE', 'SGC', 'GAT', 'Transformer']:
            from GNN import pipe
        elif config.model_name in ['CIVAE']:
            from CIVAE import pipe
        elif config.model_name in ['DMGI']:
            from DMGI import pipe
        pipe(config)
        