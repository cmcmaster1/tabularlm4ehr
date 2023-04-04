# Adapting Pretrained Language Models For Solving Tabular Prediction Problems In The Electronic Health Record

This repository contains the code for the paper [Adapting Pretrained Language Models For Solving Tabular Prediction Problems In The Electronic Health Record](chrismcmaster.com).

## Instructions

1. First you will need to generate the data by cloning https://github.com/nliulab/mimic4ed-benchmark and following the instructions up until the end of step 2. At this point you will have *train.csv* and *test.csv*
2. Place the *train.csv* and *test.csv* files in the *data/source* directory
3. Run the scripts found in tabular_lm/data to generate the datasets
4. Train the models using the finetune.py script (must specify which dataset using the --task argument)