import wandb
import torch
import logging
import torch.nn as nn
from tqdm import tqdm    
from torchvision.transforms import GaussianBlur
from bsd_dataset.common.metrics import rmse, bias, pearsonr
import math

def get_metrics(model, dataloader, prefix, options):
    metrics = {}

    model.eval()

    total_rmse = 0
    total_bias = 0
    total_pearsonr = 0

    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)
            target = target.nan_to_num()
            predictions = model(context)
            for index in range(len(context)): # len(context) = 16
                if (~mask[index]).sum() != 0:
                    num_samples += 1
                    total_rmse += rmse(predictions[index], target[index], mask[index])
                    total_bias += bias(predictions[index], target[index], mask[index])
                    if not math.isnan(pearsonr(predictions[index], target[index], mask[index])):
                        total_pearsonr += pearsonr(predictions[index], target[index], mask[index])
                    # the pearsonr could be nan for a particular patch since one of the target patch could be all 0, although the entire grid cannot be all 0

        total_rmse /= num_samples
        total_bias /= num_samples
        total_pearsonr /= num_samples

        metrics[f"{prefix}_rmse"] = total_rmse
        metrics[f"{prefix}_bias"] = total_bias
        metrics[f"{prefix}_pearson_r"] = total_pearsonr

    return metrics

def evaluate(epoch, model, dataloaders, options):
    metrics = {}
    
    if(options.master):
        if(dataloaders["val"] is not None or dataloaders["test"] is not None):
            logging.info(f"Starting epoch {epoch} evaluation")

        if(dataloaders["val"] is not None): 
            metrics.update(get_metrics(model, dataloaders["val"], "val", options))
            
        if(dataloaders["test"] is not None): 
            metrics.update(get_metrics(model, dataloaders["test"], "test", options))
        
        if(metrics):
            logging.info(f"Epoch {epoch} evaluation results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics
