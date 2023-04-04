from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, utils
from datasets import load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

label2id = {'Yes': 0, 'No': 1}
id2label = {0: 'Yes', 1: 'No'}
device = "cpu"
model_name = "models/mimic_iv_task_1_descriptivecolumns_newdata"
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True).to(device)
model.eval()
model.zero_grad()
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token added to the end of the text.
cls_token_id = tokenizer.cls_token_id

dataset = load_from_disk("data/mimic_iv/mimic_iv_task_1_descriptivecolumns_newdata")


def tokenize_splits(string):
    # Create a list of tuples
    splits = []
    # Loop through the split on semicolons
    for split in string.split(';'):
        # Split on the first colon
        split = split.split(':', 1)
        split[0] = split[0] + ":"
        # Add the split to the list of tuples
        splits.append(split)
    # Return the list of tuples
    # Create a list of tuples
    token_list = []
    # Create list of masks
    token_masks = []
    # Loop through the splits
    for i in range(len(splits)):
        # Tokenize the column name
        column_name = tokenizer.encode(splits[i][0], add_special_tokens=False)
        # Tokenize the value
        value = tokenizer.encode(splits[i][1], add_special_tokens=False)
        # Tokenize a semicolon
        semicolon = tokenizer.encode("; ", add_special_tokens=False)
        # Add the column name and value to the list of tuples
        # Concatenate the column name and value
        tokens = column_name + value + semicolon
        token_list += tokens
        # Add the mask
        name_mask = [f"name_{i}"] * len(column_name)
        value_mask = [f"value_{i}"] * len(value)
        token_masks += name_mask + value_mask + ["punct"]

    # Add special tokens
    token_list = [cls_token_id] + token_list + [sep_token_id]
    
    token_masks = ["cls"] + token_masks + ["sep"]

    # Return the list of tuples
    return token_list, token_masks

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    return output.logits, output.attentions

def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = model(inputs_embeds=inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    pred = pred[position]
    return pred.max(1).values

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id]  + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)

def construct_input_ref_token_type_pair(input_ids):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[1] * seq_len], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)
    
def construct_whole_bert_embeddings(interpretable_embedding, input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / norm_fn(attributions)
    return attributions

def visualize_attributions(model, tokenizer, text):
    input_ids, ref_input_ids = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    
    scores, output_attentions = predict(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask)
    
    interpretable_embedding = configure_interpretable_embedding_layer(model, 'deberta.embeddings.word_embeddings')
    
    input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(
        interpretable_embedding,
        input_ids, ref_input_ids,
        token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids,
        position_ids=position_ids, ref_position_ids=ref_position_ids
    )
    
    lig = LayerIntegratedGradients(squad_pos_forward_func, model.deberta.embeddings)
    
    attributions, delta = lig.attribute(
        inputs=input_embeddings, 
        baselines=ref_input_embeddings, 
        additional_forward_args=(token_type_ids, position_ids,attention_mask, 0),
        return_convergence_delta=True)
    
    attributions = summarize_attributions(attributions)
    
    vis = viz.VisualizationDataRecord(
                        attributions,
                        torch.max(torch.softmax(scores[0], dim=0)),
                        torch.argmax(scores),
                        torch.argmax(scores),
                        "Admitted",
                        attributions.sum(),       
                        all_tokens,
                        delta)
    
    viz.visualize_text([vis])