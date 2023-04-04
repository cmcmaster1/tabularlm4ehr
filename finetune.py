import os
import timeit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_from_disk, DatasetDict
from tabular_lm.finetune_options import get_args
from sklearn import metrics as m
import numpy as np
import pandas as pd
import wandb
from transformers import set_seed

set_seed(seed=17)
data_dir = "data"

def main():
    opt = get_args()

    # Split the task name on _ to get the data features
    opt.data_features = opt.task.split("_")

    # If op.model contains / then split on / and take the last element
    if "/" in opt.model:
        opt.model_name = opt.model.split("/")[-1]
    else:
        opt.model_name = opt.model

    opt.output_dir = opt.task + "_" + opt.model_name

    if opt.wandb_run_name is None:
        opt.wandb_run_name = opt.task + "_" + opt.model_name

    wandb.init(project=opt.wand_project, 
           name=opt.wandb_run_name,
           group=opt.model_name,
           tags=opt.data_features
           )
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(opt.model, cache_dir=opt.cache_dir)

    # load dataset
    dataset = load_from_disk(os.path.join(opt.data_dir, opt.task))
    # If dataset is not a DatasetDict then we need to do a train/test split with seed 901
    if not isinstance(dataset, DatasetDict):
        dataset = dataset.train_test_split(test_size=0.1, seed=901)
    else:
        test = dataset["test"]
        dataset = dataset["train"].train_test_split(test_size=0.01, seed=901)
        

    # Tokenize the dataset
    def pre_tokenize(example):
        example["length"] = len(tokenizer(example["text"], truncation=False)['input_ids'])
        return example
    
    dataset = dataset.map(pre_tokenize, batched=False)
    # Get the max length of the tokenized dataset
    train_max_length = max(dataset["train"]["length"])
    test_max_length = max(dataset["test"]["length"])
    max_length = max(train_max_length, test_max_length)

    # if max_length > 400:
    #     opt.per_device_train_batch_size = opt.per_device_train_batch_size // 1.25
    #     opt.per_device_eval_batch_size = opt.per_device_eval_batch_size // 1.25

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    test = test.map(tokenize_function, batched=True)
    # If dataset contains column "target", then rename it to "labels"
    if "target" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_columns({"target": "labels"})
        test = test.rename_columns({"target": "labels"})

    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_columns({"label": "labels"})
        test = test.rename_columns({"label": "labels"})

    # Use a map function to change the labels to integers (No = 0, Yes = 1)
    def label_to_int(examples):
        examples["labels"] = 0 if examples["labels"] == "No" else 1
        return examples

    tokenized_datasets = tokenized_datasets.map(label_to_int)
    test = test.map(label_to_int)

    # Define the metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to probabilities
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predictions = probs.argmax(axis=1)
        acc = m.accuracy_score(labels, predictions)
        auc = m.roc_auc_score(labels, probs[:, 1])
        recall = m.recall_score(labels, predictions)
        precision = m.precision_score(labels, predictions)
        pr_auc = m.average_precision_score(labels, probs[:, 1], pos_label=1)

        fpr, tpr, thresholds = m.roc_curve(labels, probs[:, 1], pos_label=1)
        j_statistic = tpr - fpr
        ix = np.argmax(j_statistic)
        best_thresh = thresholds[ix]
        best_sensitivity = tpr[ix]
        best_specificity = 1 - fpr[ix]
        return {
            "accuracy": acc,
            "auc": auc,
            "recall": recall,
            "precision": precision,
            "pr_auc": pr_auc,
            "best_thresh": best_thresh,
            "best_sensitivity": best_sensitivity,
            "best_specificity": best_specificity
        }


    # define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join("models", opt.output_dir),
        num_train_epochs=opt.epochs,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        learning_rate=opt.learning_rate,
        optim=opt.optimizer_name,
        save_total_limit=2,
        report_to="wandb",
        logging_steps=opt.logging_steps,
        eval_steps=opt.eval_steps,
        save_steps=opt.save_steps,
        logging_strategy='steps',
        evaluation_strategy="steps",
        save_strategy='steps',
        tf32=True,
        run_name=opt.output_dir,
        # load best model at the end of training
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        lr_scheduler_type="constant"
    )

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # train model
    trainer.train()

    trainer.evaluate(test, metric_key_prefix="test")

    # Bootstrap evaluation on test set to obtain confidence intervals:
    # First create 50 bootstrap samples of the test set, each with 1000 samples (with replacement)
    # Then evaluate each of these 50 samples and save the results
    # Finally, compute the mean and standard deviation of the results
    # This is a very simple implementation of bootstrapping, but it should be sufficient for our purposes

    # Create 50 bootstrap samples of the test set
    if opt.n_bootstraps > 0:
        rng = np.random.default_rng(567)
        
        bootstrapped_results = []
        for i in range(opt.n_bootstraps):
            print("Bootstrap sample {} of {}".format(i + 1, opt.n_bootstraps))
            idx = rng.integers(0, len(test) - 1, len(test))
            eval = trainer.evaluate(test.select(idx), metric_key_prefix="bootstrap")
            bootstrapped_results.append(eval)

        # convert list of dicts to dict of lists
        bootstrapped_results = {k: [dic[k] for dic in bootstrapped_results] for k in bootstrapped_results[0]}
        # compute mean and 95% confidence interval
        bootstrapped_results = {k: {"mean": np.mean(v), "lci": np.percentile(v, 2.5), "uci": np.percentile(v, 97.5)} for k, v in bootstrapped_results.items()}
        # Print the results
        print("Bootstrapped results:")
        print("Accuracy: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_accuracy"]["mean"], bootstrapped_results["bootstrap_accuracy"]["lci"], bootstrapped_results["bootstrap_accuracy"]["uci"]))
        print("AUC: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_auc"]["mean"], bootstrapped_results["bootstrap_auc"]["lci"], bootstrapped_results["bootstrap_auc"]["uci"]))
        print("Recall: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_recall"]["mean"], bootstrapped_results["bootstrap_recall"]["lci"], bootstrapped_results["bootstrap_recall"]["uci"]))
        print("Precision: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_precision"]["mean"], bootstrapped_results["bootstrap_precision"]["lci"], bootstrapped_results["bootstrap_precision"]["uci"]))
        print("PR AUC: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_pr_auc"]["mean"], bootstrapped_results["bootstrap_pr_auc"]["lci"], bootstrapped_results["bootstrap_pr_auc"]["uci"]))
        print("Best sensitivity: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_best_sensitivity"]["mean"], bootstrapped_results["bootstrap_best_sensitivity"]["lci"], bootstrapped_results["bootstrap_best_sensitivity"]["uci"]))
        print("Best specificity: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(bootstrapped_results["bootstrap_best_specificity"]["mean"], bootstrapped_results["bootstrap_best_specificity"]["lci"], bootstrapped_results["bootstrap_best_specificity"]["uci"]))
        wandb.log(
            {
                "Accuracy (mean)": bootstrapped_results["bootstrap_accuracy"]["mean"],
                "Accuracy (lci)": bootstrapped_results["bootstrap_accuracy"]["lci"],
                "Accuracy (uci)": bootstrapped_results["bootstrap_accuracy"]["uci"],
                "AUC (mean)": bootstrapped_results["bootstrap_auc"]["mean"],
                "AUC (lci)": bootstrapped_results["bootstrap_auc"]["lci"],
                "AUC (uci)": bootstrapped_results["bootstrap_auc"]["uci"],
                "Recall (mean)": bootstrapped_results["bootstrap_recall"]["mean"],
                "Recall (lci)": bootstrapped_results["bootstrap_recall"]["lci"],
                "Recall (uci)": bootstrapped_results["bootstrap_recall"]["uci"],
                "Precision (mean)": bootstrapped_results["bootstrap_precision"]["mean"],
                "Precision (lci)": bootstrapped_results["bootstrap_precision"]["lci"],
                "Precision (uci)": bootstrapped_results["bootstrap_precision"]["uci"],
                "PR AUC (mean)": bootstrapped_results["bootstrap_pr_auc"]["mean"],
                "PR AUC (lci)": bootstrapped_results["bootstrap_pr_auc"]["lci"],
                "PR AUC (uci)": bootstrapped_results["bootstrap_pr_auc"]["uci"],
                "Best sensitivity (mean)": bootstrapped_results["bootstrap_best_sensitivity"]["mean"],
                "Best sensitivity (lci)": bootstrapped_results["bootstrap_best_sensitivity"]["lci"],
                "Best sensitivity (uci)": bootstrapped_results["bootstrap_best_sensitivity"]["uci"],
                "Best specificity (mean)": bootstrapped_results["bootstrap_best_specificity"]["mean"],
                "Best specificity (lci)": bootstrapped_results["bootstrap_best_specificity"]["lci"],
                "Best specificity (uci)": bootstrapped_results["bootstrap_best_specificity"]["uci"],
            }
        )

        # Log ROC curve
        logits, labels, metrics = trainer.predict(test)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        roc_curve = wandb.plot.roc_curve(labels, probs, labels = ["No", "Yes"], classes_to_plot = [1])
        pr_curve = wandb.plot.pr_curve(labels, probs, labels = ["No", "Yes"], classes_to_plot = [1])
        wandb.log({
            "ROC curve": roc_curve,
            "PR curve": pr_curve
            })
        
        # Log the labels and probs as a table
        labels_probs = pd.DataFrame({"labels": labels, "probs": probs[:, 1]})
        wandb.log({"labels_probs": wandb.Table(dataframe=labels_probs)})

        # Test inference time on CPU using a pipeline
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu")
        print("Inference time on CPU: {} seconds".format(timeit.timeit(lambda: pipe(dataset["train"][0]["text"]), number=1000)))

    else:
        # Log ROC curve
        logits, labels, metrics = trainer.predict(test)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        roc_curve = wandb.plot.roc_curve(labels, probs, labels = ["No", "Yes"], classes_to_plot = [1])
        pr_curve = wandb.plot.pr_curve(labels, probs, labels = ["No", "Yes"], classes_to_plot = [1])
        wandb.log({
            "ROC curve": roc_curve,
            "PR curve": pr_curve
            })
        
        # Log the labels and probs as a table
        labels_probs = pd.DataFrame({"labels": labels, "probs": probs[:, 1]})
        wandb.log({"labels_probs": wandb.Table(dataframe=labels_probs)})
        # Test inference time on CPU using a pipeline
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu")
        print("Inference time on CPU: {} seconds".format(timeit.timeit(lambda: pipe(dataset["train"][0]["text"]), number=1000)))


    



if __name__ == "__main__":
    main()

# Example bash script for running finetune.py:
# python finetune.py --task_name "mimic3" --model "microsoft/deberta-base" --output_dir "/data/gpfs/projects/punim1509/clinical_t5/deberta/mimic3" --cache_dir "/data/gpfs/projects/punim1509/clinical_t5/deberta/mimic3" --epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 1e-5 --optimizer_name "adamw"