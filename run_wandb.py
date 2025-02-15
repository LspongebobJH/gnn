import wandb
import yaml
from argparse import ArgumentParser

from utils import set_random_seed, SINGLE_MODALITY_MODELS, FUSE_SINGLE_MODALITY_MODELS, FUSE_SINGLE_MODALITY_MODELS_NOSIA

project_name = 'graph_test'

def modify_project_name(new_project_name):
    global project_name
    project_name = new_project_name

def main(seed=0, config=None):
    set_random_seed(seed)
    if config is not None:
        wandb.init(project=project_name, config=config)
    else:
        wandb.init(project=project_name)
    config = wandb.config
    if config.model_name in \
        ['MHGCN', 'NeuroPath', 'Mew', 'MewCustom', 'MewFuseGraph', 'MHGCNFuseGraph'] \
            + SINGLE_MODALITY_MODELS \
            + FUSE_SINGLE_MODALITY_MODELS \
            + FUSE_SINGLE_MODALITY_MODELS_NOSIA:
        from GNN import pipe
    elif config.model_name in ['CIVAE']:
        from CIVAE import pipe
    elif config.model_name in ['DMGI']:
        from DMGI import pipe
    pipe(config)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--wandb', type=str, choices=['sweep', 'normal', 'repeat'])
    parser.add_argument('--config', type=str, default='./configs/GNN.yaml')
    parser.add_argument('--project_name', type=str, default='graph_test')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
    parser.add_argument('--load_checkpoint', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    modify_project_name(args.project_name)
    config['seed'] = args.seed
    config['save_checkpoint'] = args.save_checkpoint
    config['load_checkpoint'] = args.load_checkpoint
    config['checkpoint_path'] = args.checkpoint_path

    if args.wandb == 'sweep':
        sweep_id = wandb.sweep(sweep=config, project=project_name)
        print(f"sweep id: {sweep_id}")
        wandb.agent(sweep_id, function=main)
    elif args.wandb == 'normal':
        main(args.seed, config=config)
    elif args.wandb == 'repeat':
        for seed in range(5):
            main(seed, config=config)
