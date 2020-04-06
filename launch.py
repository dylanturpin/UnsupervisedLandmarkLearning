import os
import hydra
from hydra import slurm_utils

@hydra.main(config_path='conf/config.yaml')
def main(config):
    slurm_utils.symlink_hydra(config, os.getcwd())

if __name__ == "__main__":
    main()
