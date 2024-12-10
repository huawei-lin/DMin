import os

import torch.multiprocessing as mp
import argparse
import torch
import random
import numpy as np
from DMIN.utils import get_config
from DMIN.engine import calc_infl_mp
from datasets import load_dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

CONFIG_PATH = "./config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=CONFIG_PATH, type=str)
    args = parser.parse_args()
    config_path = args.config_path

    config = get_config(config_path)
    print(config)

    random.seed(int(config.influence.seed))
    np.random.seed(int(config.influence.seed))

    calc_infl_mp(config)
    print("Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
