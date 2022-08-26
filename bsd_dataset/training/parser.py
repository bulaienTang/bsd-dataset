import os
import argparse
from .utils import root

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type = str, default = "default", help = "Experiment name")
    parser.add_argument("--seed", type = int, default = 0, help = "Seed")
    parser.add_argument("--logs", type = str, default = os.path.join(root, "logs/"), help = "Logs path")
    parser.add_argument("--model", type = str, default = None, help = "Model name")
    parser.add_argument("--model_config", type = str, default = None, help = "Path to model config file")
    parser.add_argument("--no_train", action = "store_true", default = False, help = "No training")
    parser.add_argument("--no_val", action = "store_true", default = False, help = "No validation")
    parser.add_argument("--no_test", action = "store_true", default = False, help = "No test")
    parser.add_argument("--eval_freq", type = int, default = 1, help = "Evaluation frequency")
    parser.add_argument("--data", type = str, default = None, help = "Path to the data directory")
    parser.add_argument("--device", type = str, default = None, choices = ["cpu", "gpu"], help = "Specify device (gpu > cpu)")
    parser.add_argument("--device_id", type = int, default = 0, help = "Specify device id for single gpu training")
    parser.add_argument("--distributed", action = "store_true", default = False, help = "Use multiple gpus if available")
    parser.add_argument("--backend", type = str, default = "nccl", help = "Distributed backend")
    parser.add_argument("--address", type = str, default = "127.0.0.1", help = "Distributed TCP address")
    parser.add_argument("--port", type = str, default = 6100, help = "Distributed TCP port")
    parser.add_argument("--device_ids", nargs = "+", default = None, help = "Specify device ids for multi gpu training")
    parser.add_argument("--wandb", action = "store_true", default = False, help = "Enable wandb logging")
    parser.add_argument("--notes", type = str, default = None, help = "Notes for experiment")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers")
    parser.add_argument("--epochs", type = int, default = 16, help = "Number of train epochs")
    parser.add_argument("--batch_size", type = int, default = 16, help = "Batch size")
    parser.add_argument("--lr", type = float, default = 5e-4, help = "Learning rate")
    parser.add_argument("--beta1", type = float, default = 0.9, help = "Adam momentum factor (Beta 1)")
    parser.add_argument("--beta2", type = float, default = 0.999, help = "Adam rmsprop factor (Beta 2)")
    parser.add_argument("--eps", type = float, default = 1e-8, help = "Adam eps")
    parser.add_argument("--weight_decay", type = float, default = 0.01, help = "Adam weight decay")
    parser.add_argument("--num_warmup_steps", type = int, default = 100, help = "Number of steps to warmup the learning rate")
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

    options = parser.parse_args()
    return options
