# maybe https://www.kaggle.com/lezwon/parallel-kfold-training-on-tpu-using-pytorch-li
import argparse
import numpy as np
import os
import pytorch_lightning as pl
import random
import string
import torch
import wandb
import yaml
import hydra
from omegaconf import OmegaConf

from notebooks.utils import custom_layers
import src.modules

from pytorch_lightning.loggers import WandbLogger

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from py3nvml import *
from sklearn.model_selection import StratifiedKFold
from src.datasets import SimpleDataset
from src.trainer import ConnectivityNetworksClassifier
from src.utils import Builder, build_network, load_yaml
from torch.utils.data import Subset


def get_free_gpu():
    py3nvml.nvmlInit()
    device_count = py3nvml.nvmlDeviceGetCount()
    memories = []
    for i in range(device_count):
        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)  # Need to specify GPU
        mem = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        memories.append(mem.free)
    return np.argmax(memories)


def make_best_metrics(run_name):
    VAL_ACC_KEY = "val_acc"
    VAL_LOSS_KEY = "val_loss"

    api = wandb.Api()

    run = api.run(run_name)
    history = run.scan_history()

    val_losses = [row[VAL_LOSS_KEY] for row in history if VAL_LOSS_KEY in row]
    val_accs = [row[VAL_ACC_KEY] for row in history if VAL_ACC_KEY in row]

    run.summary[f"best_{VAL_LOSS_KEY}"] = min(val_losses)
    run.summary[f"best_{VAL_ACC_KEY}"] = max(val_accs)

    run.update()


def run_process(params):
    config = deepcopy(params["config"])
    OmegaConf.set_struct(config, False)
    dataset = params["dataset"]

    train_idx = params["train_idx"]
    val_idx = params["val_idx"]

    idx_split = params["idx_split"]
    idx_run = params["idx_run"]
    random_state_split = params["random_state_split"]
    random_state_run = params["random_state_run"]

    np.random.seed(random_state_run)
    if random_state_run is not None:
        torch.manual_seed(random_state_run)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    config.run_info = {
        "idx_split": int(idx_split), "idx_run": int(idx_run),
        "random_state_split": int(random_state_split), "random_state_run": int(random_state_run)
    }
    # TODO fix bicycle
    config.opt = {
        "lr": config.train.lr,
        "name": config.train.opt_name
    }
    
    builder = Builder(torch.nn.__dict__, src.modules.__dict__, custom_layers.__dict__)
    net = build_network(config.network.architecture, builder).double()

    experiment = wandb.init(
        entity=config.wandb_params.entity,
        project=config.wandb_params.project,
        group=config.wandb_params.group,
        reinit=True
    )
    wandb_logger = WandbLogger(
        experiment=experiment
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(config))
    OmegaConf.save(config, os.path.join(wandb_logger.experiment.dir, config.wandb_params.config_name))

    opt_kwargs = deepcopy(config.opt)
    opt_name = opt_kwargs.pop("name")
    opt_lr = opt_kwargs.pop("lr")

    model = ConnectivityNetworksClassifier(net, train_dataset, val_dataset,
                                           opt=opt_name, opt_lr=opt_lr, opt_kwargs=opt_kwargs,
                                           batch_size=config.train.batch_size,
                                           num_workers=1)
    # TODO make option of backend
    trainer = pl.Trainer(gpus=[get_free_gpu()], max_epochs=config.train.max_epochs,
                         logger=wandb_logger, checkpoint_callback=None, distributed_backend=None)
    trainer.fit(model)
    experiment.save()
    experiment.finish()
    make_best_metrics(f"{config.wandb_params.entity}/{config.wandb_params.project}/{experiment.id}")


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
    dataset = SimpleDataset(hydra.utils.to_absolute_path(config.dataset.data_path),
                            hydra.utils.to_absolute_path(config.dataset.targets_path))
    wandb_group = config.wandb_params.group
    if wandb_group is None:
        config.wandb_params.group = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    seed = config.setup.seed
    n_runs = config.setup.n_runs
    n_splits = config.setup.n_splits

    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    kfold_splits = []
    split_seeds = np.random.choice(np.iinfo(np.int32).max, size=n_runs)
    for i in range(n_runs):
        for split in StratifiedKFold(n_splits=n_splits, random_state=split_seeds[i], shuffle=True).split(
                dataset.data, dataset.targets):
            kfold_splits.append(split)

    process_seeds = np.random.choice(np.iinfo(np.int32).max, size=n_splits * n_runs)

    params = [{
        "config": config,
        "dataset": dataset,
        "train_idx": kfold_splits[i][0],
        "val_idx": kfold_splits[i][1],
        "idx_split": i // 10,
        "idx_run": i % 10,
        "random_state_split": split_seeds[i // 10],
        "random_state_run": process_seeds[i],
    } for i in range(n_splits * n_runs)]

    with ProcessPoolExecutor(max_workers=config.setup.num_workers) as pool:
        pool.map(run_process, params)


if __name__ == "__main__":
    main()
