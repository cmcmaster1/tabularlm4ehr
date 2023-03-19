from tabular_lm.attribution import get_dataset_attributions, split_on_semicolon
from matplotlib import pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, utils
from datasets import load_from_disk
import pandas as pd

device = "cuda"
model_name = ""
dataset_path = ""
task = 1
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

dataset = load_from_disk(dataset_path)
test_examples = dataset['test'].shuffle(seed=42).select(range(5000))

test_attributions = get_dataset_attributions(test_examples)
test_attributions_df = pd.concat(test_attributions)

all_splits = split_on_semicolon(test_examples[0]["text"])
splits = []
# Loop through the split on semicolons
for split in all_splits:
    # Split on the first colon
    split = split.split(':', 1)
    split[0] = split[0] + ":"
    # Add the split to the list of tuples
    splits.append(split)

vars = [s[0][:-1] for s in splits]
# Create a dict of the form {"value_0": vars[0], "value_1": vars[1], ...}
vars_dict = {f"value_{i}": vars[i] for i in range(len(vars))}

overall_att = test_attributions_df.groupby("token_mask").agg({"abs_max": "mean"})
plot = overall_att.loc[vars_dict.keys()].plot.bar(rot=90, figsize=(20, 10), fontsize=20, color=sns.color_palette("Set1", 1))
plot.set_xticklabels([vars_dict[label.get_text()] for label in plot.get_xticklabels()])
plot.set_ylabel("Attribution", fontsize=20)
plot.set_xlabel(None)
plot.set_title(f"Model {task}", fontsize=20)
plot.legend(["Absolute attribution (mean)"], fontsize=20)

plt.savefig(f"model{task}_attribution_abs.pdf", bbox_inches='tight')