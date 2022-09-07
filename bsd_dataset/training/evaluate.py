from bsd_dataset.training import data
import wandb
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm    
from torchvision.transforms import GaussianBlur
from bsd_dataset.common.metrics import rmse, bias, pearsonr
import math

def get_metrics(model, num_patches, dataloader, prefix, options):
    metrics = {}

    model.eval()

    total_rmse = 0
    total_bias = 0
    total_pearsonr = 0

    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # context_shape: torch.Size([16, 1, 15, 36]), 
            # target_shape: torch.Size([16, 80, 200]), 
            # patch_prediction_shape: torch.Size([16, 16, 100])
            # predictions: torch.Size([10, 16, 16, 100])
            # predictions_folded: torch.Size([16, 80, 200])
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)
            target = target.nan_to_num()
            predictions = torch.Tensor(()).to(options.device) 

            # split the test/val x into ten patches, run through the model and concatenate the result predictions
            o = F.unfold(context, kernel_size=(3,18), stride=(3,18))
            o = o.view(context.shape[0], context.shape[1], 3, 18, num_patches) # 16, 1, 3, 18, 10
            o = o.permute(4, 0, 1, 2, 3) # 10, 16, 1, 3, 18

            for patch in o:
                prediction = model(patch).unsqueeze(0).to(options.device)
                predictions = torch.cat((predictions, prediction),0).to(options.device) 
                # 10, 16, 16, 100

            # transform the predictions into the format of the target output
            predictions = predictions.unsqueeze(0).permute(0, 2, 3, 4, 1) 
            #1, 10, 16, 16, 100 => 1, 16, 16, 100, 10
            predictions = predictions.view(1, len(context)*target.shape[1]*target.shape[2], num_patches) # 1, 16*16*100, 10
            predictions = F.fold(predictions, output_size=(80, 200), kernel_size=(16,100), stride=(16,100)).squeeze(0) # 16, 80, 200

            for index in range(len(context)): # len(context) = 16
                if (~mask[index]).sum() != 0:
                    num_samples += 1
                    total_rmse += rmse(predictions[index], target[index], mask[index])
                    total_bias += bias(predictions[index], target[index], mask[index])
                    # if not math.isnan(pearsonr(predictions[index], target[index], mask[index])):
                    total_pearsonr += pearsonr(predictions[index], target[index], mask[index])
                    # the pearsonr could be nan for a particular patch since one of the target patch could be all 0, although the entire grid cannot be all 0

        total_rmse /= num_samples
        total_bias /= num_samples
        total_pearsonr /= num_samples

        metrics[f"{prefix}_rmse"] = total_rmse
        metrics[f"{prefix}_bias"] = total_bias
        metrics[f"{prefix}_pearson_r"] = total_pearsonr

    return metrics

def evaluate(epoch, model, valTestDataloaders, num_patches, options):
    metrics = {}

    if(options.master):
        if(valTestDataloaders["val"] is not None or valTestDataloaders["test"] is not None):
            logging.info(f"Starting epoch {epoch} evaluation")

        if(valTestDataloaders["val"] is not None): 
            metrics.update(get_metrics(model, num_patches, valTestDataloaders["val"], "val", options))
            
        if(valTestDataloaders["test"] is not None): 
            metrics.update(get_metrics(model, num_patches, valTestDataloaders["test"], "test", options))
        
        if(metrics):
            logging.info(f"Epoch {epoch} evaluation results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics
