import numpy as np
from sagemaker import get_execution_role
import boto3
import pandas as pd
from io import StringIO # Python 3.
from datasets import load_dataset,Dataset,DatasetDict,concatenate_datasets

from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import os
from tqdm import tqdm
import string
from itertools import chain

# import our .py files with select functions
import dataset
from tokenizer import encode_with_truncation, encode_without_truncation, group_texts, column_to_files
from utility import train_validate_test_split

# load data
s3 = boto3.resource('s3')
bucket_name = 'chianglab-dataderivatives'
file_path = "mimic-iv-ed-2.2/text_repr.json"

# loading in raw data
content_object = s3.Object(bucket_name, file_path)
file_content = content_object.get()['Body'].read().decode('utf-8')
json_content = json.loads(file_content)
df = pd.DataFrame(json_content).T

# fill in missing entries
df['medrecon'] = df['medrecon'].fillna("The patient was previously not taking any medications.")
df['pyxis'] = df['pyxis'].fillna("The patient did not receive any medications.")
df['vitals'] = df['vitals'].fillna("The patient had no vitals recorded")
df['codes'] = df['codes'].fillna("The patient received no diagnostic codes")

# print 70% and 30% of splits
train, validate, test = train_validate_test_split(df, seed=7)
train2, validate2, test2 = train_validate_test_split(df, seed=7)
print("70% Train:",len(train), "\n30% Test:",len(validate+test))

#### RECORD PATIENT INFORMATION ####

# extracts the patient ID's from the arrival column 
train_patients = train.arrival.astype(str).str.split().str[1].to_list()
train_patients2 = train2.arrival.astype(str).str.split().str[1].to_list()
test_patients = test.arrival.astype(str).str.split().str[1].to_list()
validate_patients = validate.arrival.astype(str).str.split().str[1].to_list()
test_patients = (test_patients+validate_patients)

# Sanity Check: checking if seed works by seeing if training sets are equal when called two separate times for future reproducibility
train_patients.sort()
train_patients2.sort()

train_patients = [''.join(char for char in item if char not in string.punctuation) for item in train_patients]
train_patients2 = [''.join(char for char in item if char not in string.punctuation) for item in train_patients2]
test_patients = [''.join(char for char in item if char not in string.punctuation) for item in test_patients]
 
# using == to check if lists are equal
if train_patients == train_patients2:
    print("The lists are identical")
else:
    print("The lists are not identical")

# free up memory by deleting
del train2
del validate2
del test2

# remove duplicates    
train_patients = set(train_patients)
test_patients = set(test_patients)

# write patient ID's into txt files for lookup purposes later in case there are dependency issues in the future that modify seeding
file = open('./models/data/train_patients.txt','w')
for patient in train_patients:
	file.write(patient+"\n")
file.close()
print("stored patient IDs into: ./models/data/train_patients.txt")

file = open('./models/data/test_patients.txt','w')
for patient in test_patients:
	file.write(patient+"\n")
file.close()
print("stored patient IDs into: ./models/data/test_patients.txt")


# feed it into a custom tokenizer but first need to make a Dataset Object for transformers
disposition_train = train.eddischarge_category
train = train.drop("eddischarge_category",axis=1)
stacked_train = train.stack().to_frame("headline")
print("Train Stacked")

disposition_test = test.eddischarge_category
test = test.drop("eddischarge_category",axis=1)
stacked_test = test.stack().to_frame("headline")
print("Test Stacked")

training_data_corpus = Dataset.from_pandas(stacked_train)
testing_data_corpus = Dataset.from_pandas(stacked_test)

txt_files_dir = "./models/data/text_split/"
os.system("rm -rf {txt_files_dir}")
os.system("mkdir {txt_files_dir}")

# Get the training data
training_data = stacked_train["headline"]
# Removing the end of line character \n
training_data = training_data.replace("\n"," ")
# Create a file for every description value
train_num_files = column_to_files(training_data, txt_files_dir, output_filename="train.txt")
print("Turned train dataset into txt file")

tokenizer_dir = "./models/data/TokenizerRoBERTa"
os.system("rm -rf {tokenizer_dir}")
os.system("mkdir {tokenizer_dir}")

# Train Tokenizer
paths = [str(x) for x in Path(".").glob("./models/data/text_split/*.txt")]
print("Loaded Dataset:", str(len(paths)))
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

print("Train")
# Customize training

# we choose a vocab_size of 30,522 to reduce the OOV tokens which may commonly be found in Medical Terminology
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model(tokenizer_dir)

os.system("rm -rf {txt_files_dir}")

# Create the tokenizer using vocab.json and mrege.txt files
tokenizer = ByteLevelBPETokenizer(
    os.path.abspath(os.path.join(tokenizer_dir,'vocab.json')),
    os.path.abspath(os.path.join(tokenizer_dir,'merges.txt'))
)
# Prepare the tokenizer
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# Set a configuration for our RoBERTa model
config = RobertaConfig(
    vocab_size=30522,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config)
print('Num parameters: ',model.num_parameters())

# Create the tokenizer from our trained one
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_len=512)

# Tokenize training set
truncate = False
if truncate:
    print("Tokenizing with truncation")
    train_data_tokenized = training_data_corpus.map(encode_without_truncation, batched=True)
    train_data_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("Tokenizing Complete")
else:
    print("Tokenizing without truncation")
    train_data_tokenized = training_data_corpus.map(encode_without_truncation, batched=True)
    train_data_tokenized.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    print("Training Tokenizing Complete")
    
# Tokenize testing set
if truncate:
    print("Tokenizing with truncation")
    train_data_tokenized = training_data_corpus.map(encode_without_truncation, batched=True)
    train_data_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("Tokenizing Complete")
else:
    print("Tokenizing without truncation")
    test_data_tokenized = testing_data_corpus.map(encode_without_truncation, batched=True)
    test_data_tokenized.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    print("Testing Tokenizing Complete")

# Since no truncation was involved we have to make all vectors 512
max_length = 512
if not truncate:
    train_dataset = train_data_tokenized.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
    # convert them from lists to torch tensors
    train_dataset.set_format("torch")
    test_dataset = test_data_tokenized.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
    # convert them from lists to torch tensors
    test_dataset.set_format("torch")

print("data preprocessing is finished")

# to feed into training
eval_dataset = test_dataset

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

TRAIN_BATCH_SIZE = 32    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 32    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 5        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-4    # learning rate (default: 0.001)
WEIGHT_DECAY = 0.01
SEED = 42               # random seed (default: 42)
MAX_LEN = 128
SUMMARY_LEN = 7

model_dir = "./models/EHR-RoBERTa"
os.system("rm -rf {model_dir}")
os.system("mkdir {model_dir}")

print(model_dir)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    evaluation_strategy = 'steps',
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    logging_steps=5000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=5000,
    #eval_steps=4096,
    save_total_limit=1,
)
# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #prediction_loss_on

# train
trainer.train()