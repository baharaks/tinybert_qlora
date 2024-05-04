
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:49:26 2024

@author: becky
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    BertConfig, BertForSequenceClassification, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, DataCollatorWithPadding, logging, pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset

torch.cuda.empty_cache()

# Model from Hugging Face hub
base_model = "huawei-noah/TinyBERT_General_4L_312D"

# Load the dataset using pandas
data = pd.read_csv('data/BBC News Train.csv')
ArticleId = data['ArticleId']
Category = data['Category']
data['label'], unique = pd.factorize(data['Category'])
new_data = data[['Text', 'label']]

test_data = pd.read_csv("data/BBC News Test.csv")

new_test_data = test_data 

# Verify that labels are within the expected range
num_classes = len(unique)
assert all(new_data['label'].between(0, num_classes - 1)), "Invalid label values in the dataset"

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(new_data)
test_dataset = Dataset.from_pandas(new_test_data)

compute_dtype = getattr(torch, "float16")

# Configure quantization 4 bits
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Initialize model with updated configuration
config = BertConfig.from_pretrained(base_model)
config.max_position_embeddings = 2048 #512  # Adjusted for TinyBERT's maximum length
config.num_labels = num_classes  # Set the number of labels based on the dataset
model = BertForSequenceClassification(config)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new PAD token

tokenizer.padding_side = "right"

# Function to truncate input sequences and ensure device transfer
def truncate_and_transfer(example):
    tokens = tokenizer(example['Text'], padding='max_length', truncation=True, max_length=512)
    input_ids = torch.tensor(tokens['input_ids']).to(device)
    attention_mask = torch.tensor(tokens['attention_mask']).to(device)
    label = torch.tensor(example['label']).to(device)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

# Apply truncation and device transfer to dataset
dataset = dataset.map(truncate_and_transfer)

# Data collator to ensure batch sizes match
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# LoRA configuration
peft_params = LoraConfig(
    task_type="SEQ_CLS",  # Ensure correct task type for sequence classification
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# Training Arguments
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=500,
    per_device_train_batch_size=16,  # Adjust batch size as per your resources
    gradient_accumulation_steps=1,
    optim="adamw_hf",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
)

# Initialize trainer with corrected settings
trainer = SFTTrainer(
    model=model,
    args=training_params,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    peft_config=peft_params,
    dataset_text_field="input_ids"  # Set to the input IDs field after truncation
)

# Train the model
trainer.train()

new_model = "TinyBERT_General_4L_312D_QLoRa"
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

from tensorboard import notebook
log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))

logging.set_verbosity(logging.CRITICAL)

# Ensure consistent device usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

# Test
test_result = []
for id in range(len(test_dataset)):
    prompt = test_dataset[id]['Text']
    label = classify_text(prompt)
    print(f"Predicted Label: {label}")
    test_result.append(label)
    
    
new_test_data['Category'] = test_result
category_map = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 5: 'entertainment'}
new_test_data['Category'] = new_test_data['Category'].map(category_map)

df = pd.DataFrame({'ArticleId': new_test_data['ArticleId'], 'Category': new_test_data['Category']})

df.to_csv('BBC News Validation.csv', index=False)
    
    
    
    
    