# Argparse options for finetuning a model using finetune.py

import argparse
import os
import sys

base_dir = "/mnt/hdd/projects/pretraining_data/multitasks"
tasks_dir = os.path.join(base_dir, "processed_data/tabular")
cache_dir = os.path.join(base_dir, "cache")

# The tasks are found as datasets in the data/tabular directory, with each in its own folder
# We can therefore list the tasks by listing the folders in the data/tabular directory
tasks = [f.name for f in os.scandir(tasks_dir) if f.is_dir()]



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_and_data", type=str, default="t5-small", help="The model and data to use.")
    parser.add_argument("--cache_dir", type=str, default=cache_dir, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for the optimizer.")
    parser.add_argument("--optimizer_name", type=str, default="adamw_torch", help="The name of the optimizer to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--n_bootstraps", type=int, default=200, help="Number of bootstraps to run.")


    options = parser.parse_args()
    return options
    