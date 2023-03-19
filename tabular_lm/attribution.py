from captum.attr import LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import pandas as pd
import torch

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

def split_on_semicolon(string):
        # Split on semicolons
        splits = string.split(';')
        # We will need to loop through the splits and add the previous split to the current one if it does not contain a colon
        new_splits = []
        for i, split in enumerate(splits):
            if ':' not in split:
                new_splits[-1] = new_splits[-1] + ';' + split
            else:
                new_splits.append(split)
        return new_splits

def tokenize_splits(string, tokenizer):
    # Create a list of tuples
    all_splits = split_on_semicolon(string)
    splits = []
    # Loop through the split on semicolons
    for split in all_splits:
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

    
    token_masks = ["cls"] + token_masks + ["sep"]

    # Return the list of tuples
    return token_list, token_masks

def tokenize(example):
    example["input_ids"], example["token_masks"] = tokenize_splits(example["text"])
    return example

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None, model=None):
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    return output.logits, output.attentions

def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0, model=None):
    pred = model(inputs_embeds=inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    pred = pred[position]
    return pred.max(1).values

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, device="cuda"):
    text_ids, token_masks = tokenize_splits(text)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id]  + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), token_masks

def construct_input_ref_token_type_pair(input_ids, device="cuda"):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[1] * seq_len], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids, device):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device="cuda")
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)
    
def construct_whole_deberta_embeddings(interpretable_embedding, input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / norm_fn(attributions)
    return attributions

def attribute(model, tokenizer, device, text, label, ref_token_id, sep_token_id, cls_token_id):
    input_ids, ref_input_ids, token_masks = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, device)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, device)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids, device)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    
    scores, output_attentions = predict(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        model=model
    )
    
    interpretable_embedding = configure_interpretable_embedding_layer(model, 'deberta.embeddings.word_embeddings')
    
    input_embeddings, ref_input_embeddings = construct_whole_deberta_embeddings(
        interpretable_embedding,
        input_ids, ref_input_ids,
        token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids,
        position_ids=position_ids, ref_position_ids=ref_position_ids
    )
    
    lig = LayerIntegratedGradients(squad_pos_forward_func, model.deberta.embeddings)
    
    attributions, delta = lig.attribute(
        inputs=input_embeddings, 
        baselines=ref_input_embeddings, 
        additional_forward_args=(token_type_ids, position_ids,attention_mask, 0, model),
        return_convergence_delta=True)
    
    attributions = summarize_attributions(attributions)
    
    scores = scores.detach().cpu()
    attributions = attributions.detach().cpu().numpy()

    # Get the max, min and max absolute attribution for each type of token mask
    df = pd.DataFrame({"attribution": attributions, "token_mask": token_masks})
    min_max = df.groupby("token_mask").agg({"attribution": ["min", "max"]})
    min_max.columns = ["_".join(x) for x in min_max.columns.ravel()]
    min_max = min_max.reset_index()
    df = df.merge(min_max, on="token_mask", how="left")
    df["abs"] = df["attribution"].abs()
    df["abs_max"] = df["token_mask"].apply(lambda x: df[df["token_mask"] == x]["abs"].max())

    # Add label
    df["label"] = label
    df["prediction"] = torch.argmax(scores).item()
    # If prediction is 1 then change to No, else Yes
    df["prediction"] = df["prediction"].apply(lambda x: "No" if x == 1 else "Yes")

    remove_interpretable_embedding_layer(model, interpretable_emb=interpretable_embedding)

    return [attributions, all_tokens, token_masks, df]

# Dataset attribution
def get_dataset_attributions(examples, model, tokenizer, device, ref_token_id, sep_token_id, cls_token_id):
    model.eval()
    model.zero_grad()
    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id
    attributions = []
    for example in examples:
        attributions.append(attribute(model, tokenizer, device, example["text"], example["label"], ref_token_id, sep_token_id, cls_token_id)[3])
    return attributions