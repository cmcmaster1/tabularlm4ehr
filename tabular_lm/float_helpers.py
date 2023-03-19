from datasets import Dataset
import numpy as np
import torch
import re

def create_tab_text(dataframe, x_cols, y_col, numeric_columns=None, norm_dataframe=None):
    dframe = dataframe.copy()
    if numeric_columns is None:
        numeric_columns = dframe.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if norm_dataframe is not None:
            dframe[col] = (dframe[col] - norm_dataframe[col].mean()) / norm_dataframe[col].std()
        else:
            dframe[col] = (dframe[col] - dframe[col].mean()) / dframe[col].std()
        dframe[col] = dframe[col].apply(lambda x: '[[' + str(x) + ']]')
    dframe['text'] = dframe[x_cols].apply(lambda x: '; '.join([f'{col}: {val}' for col, val in zip(x_cols, x)]), axis=1)
    dframe = dframe[['text', y_col]]
    dframe.rename(columns={y_col: 'labels'}, inplace=True)
    return Dataset.from_pandas(dframe)

def tokenize_with_floats(texts, tokenizer, max_seq_length=512):
    # Add "<float>" to the tokenizer's vocabulary if not already present
    if "<float>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<float>"])
    float_token_id = tokenizer.convert_tokens_to_ids("<float>")
    
    # Define a regular expression pattern to match floats enclosed in double square brackets
    pattern = r"\[\[(\d+\.\d+)\]\]"
    
    # Find all float matches in the texts
    float_matches = [re.findall(pattern, text) for text in texts]
    float_counts = [len(matches) for matches in float_matches]
    
    # Replace all float matches with a placeholder token "<float>"
    texts = [re.sub(pattern, "<float>", text) for text in texts]
    
    # Tokenize the modified texts and encode them as inputs
    tokenized_inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Convert the input_ids to a float tensor
    input_ids = tokenized_inputs["input_ids"].float()
    
    # Create a float mask tensor for each batch element
    float_masks = np.zeros((len(texts), max_seq_length), dtype=np.int64)
    for i, (matches, count) in enumerate(zip(float_matches, float_counts)):
        if count == 0:
            continue
        float_positions = np.where(input_ids[i] == float_token_id)[0][:count]
        float_values = [float(match) for match in matches]
        input_ids[i, float_positions] = torch.tensor(float_values)
        float_masks[i, float_positions] = 1
    
    # Convert float_masks to a PyTorch tensor
    float_masks = torch.tensor(float_masks)
    
    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_inputs["attention_mask"],
        "token_type_ids": tokenized_inputs["token_type_ids"],
        "float_mask": float_masks
    }

def batch_tokenize_with_floats(examples, tokenizer):
    return tokenize_with_floats(examples["text"], tokenizer)