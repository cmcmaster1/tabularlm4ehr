# Argparse options for finetuning a model using finetune.py

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="austin/mimic-pubmed-deberta-small", help="The model to use.")
    parser.add_argument("--tokenizer", type=str, default=None, help="The tokenizer to use (if different to the model).")
    parser.add_argument("--task", type=str, default="all", help="The task to finetune on.")
    parser.add_argument("--data_dir", type=str, default="data", help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--wandb_project", type=str, default="tabular_lm", help="The wandb project to use.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="The wandb run name to use.")
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
    