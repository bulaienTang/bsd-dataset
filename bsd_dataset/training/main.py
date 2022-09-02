import os
os.environ["WANDB_SILENT"] = "true"

import sys
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .parser import parse_args
from .train import train
from .evaluate import evaluate
from .data import load as load_dataloaders
from .model import load as load_model
from .optimizer import load as load_optimizer
from .scheduler import load as load_scheduler
from .logger import Logger
from .utils import seeder

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def worker(rank, options, logger):
    logger.add(rank = rank)

    if(rank == 0):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key, value in sorted(vars(options).items()):
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    options.rank = rank
    options.master = rank == 0
    
    if(options.device == "cuda"):
        options.device_id = options.device_ids[options.rank] if(options.distributed) else options.device_id
        options.device = f"cuda:{options.device_id}"

    logging.info(f"Using {options.device} device")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = f"tcp://{options.address}:{options.port}", world_size = options.nprocs, rank = options.rank)
        options.batch_size = options.batch_size // options.nprocs

    num_patches = 10
    dataloadersList, input_shape, target_shape = load_dataloaders(options, num_patches)

    model = load_model(model = options.model, input_shape = input_shape, target_shape = target_shape, model_config = options.model_config)

    if(options.device == "cpu"):
        model.float()
    else:
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])

    # train ten patches separately and record the metrics for the last epochs of each patch
    all_metrics = []

    for patch in range(num_patches):
        
        # get one patch
        dataloaders = dataloadersList[patch]

        optimizer = None
        scheduler = None
        if(dataloaders["train"] is not None):        
            optimizer = load_optimizer(model = model, lr = options.lr, beta1 = options.beta1, beta2 = options.beta2, eps = options.eps, weight_decay = options.weight_decay)
            scheduler = load_scheduler(optimizer = optimizer, base_lr = options.lr, num_warmup_steps = options.num_warmup_steps, num_total_steps = dataloaders["train"].num_batches * options.epochs)

        start_epoch = 0
        if(options.checkpoint is not None):
            if(os.path.isfile(options.checkpoint)):
                checkpoint = torch.load(options.checkpoint, map_location = options.device)
                start_epoch = checkpoint["epoch"]
                model_state_dict = checkpoint["model_state_dict"]
                if(not options.distributed and next(iter(model_state_dict.items()))[0].startswith("module")):
                    model_state_dict = {key[len("module."):]: value for key, value in model_state_dict.items()}
                model.load_state_dict(model_state_dict)
                if(optimizer is not None): 
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
            else:
                logging.info(f"No checkpoint found at {options.checkpoint}")

        if(options.wandb and options.master):
            logging.debug("Starting wandb")
            wandb.init(project = "climate-downscaling-benchmark", notes = options.notes, tags = [], config = vars(options))
            wandb.run.name = options.name
            wandb.save(os.path.join(options.log_dir_path, "params.txt"))

        # evaluate once before training
        evaluate(start_epoch, model, dataloaders, options)

        if(dataloaders["train"] is not None):
            options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
            os.makedirs(options.checkpoints_dir_path, exist_ok = True)

            scaler = GradScaler()

            best_loss = np.inf
            for epoch in range(start_epoch + 1, options.epochs + 1):
                if(options.master): 
                    logging.info(f"Starting epoch {epoch}")

                start = time.time()
                train(epoch, model, dataloaders, optimizer, scheduler, scaler, options)
                end = time.time()

                if(options.master): 
                    logging.info(f"Finished epoch {epoch} in {end - start:.3f} seconds")

                # evaluate by frequency set, not the last epoch
                if epoch % options.eval_freq == 0 & epoch != options.epochs:
                    metrics = evaluate(epoch, model, dataloaders, options)

                    if(options.master):
                        checkpoint = {"epoch": epoch, "name": options.name, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                        if("loss" in metrics):
                            if(metrics["loss"] < best_loss):
                                best_loss = metrics["loss"]
                                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))
                # last epoch
                elif epoch == options.epochs:
                    metrics = evaluate(epoch, model, dataloaders, options)

                    if(options.master):
                        checkpoint = {"epoch": epoch, "name": options.name, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                        if("loss" in metrics):
                            if(metrics["loss"] < best_loss):
                                best_loss = metrics["loss"]
                                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))
                    
                    all_metrics.append(metrics)

    ###################################################################
    # final evaluation of combined prediction against original target #
    ###################################################################
    total_rmse = 0
    total_bias = 0
    total_pearson_r = 0
    for metrics in all_metrics:
        total_rmse += metrics["test_rmse"]
        total_bias += metrics["test_bias"]
        total_pearson_r += metrics["test_pearson_r"]
    
    logging.info(f"total_test_rmse: {total_rmse}")
    logging.info(f"total_test_bias: {total_bias}")
    logging.info(f"total_test_pearson_r: {total_pearson_r}")

    ###################################################################
    # final evaluation of combined prediction against original target #
    ###################################################################

    if(options.distributed):
        dist.destroy_process_group()
    
    if(options.wandb and options.master):
        wandb.finish()

if(__name__ == "__main__"):    
    options = parse_args()

    options.worker_init_fn, options.generator = seeder(options.seed)

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger = Logger(file = options.log_file_path)

    logger.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.nprocs = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.nprocs = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.nprocs = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.nprocs = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.nprocs, args = (options, logger))
    
    logger.stop()
