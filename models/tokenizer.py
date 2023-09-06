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
from itertools import chain

import os
from tqdm import tqdm


def encode_with_truncation(examples):
  """
  Mapping function to tokenize the sentences passed with truncation
  """
  return tokenizer(examples["headline"], truncation=True, padding="max_length",
                    max_length=512, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """
  Mapping function to tokenize the sentences passed without truncation
  """
  return tokenizer(examples["headline"], return_special_tokens_mask=True)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this
def column_to_files(column, txt_files_dir,output_filename="train.txt"):
    """
    Function that converts batches of dataframe column into txt files
    """
    # The prefix is a unique ID to avoid to overwrite a text file
    i=1
    counter = 0
    #For every value in the df, with just one column
    for row in tqdm(column.to_list()):
      # Create the filename using the prefix ID
        if i % 1000 == 1:
            file_name = os.path.join(txt_files_dir, str(counter)+'.txt')
            f = open(file_name, 'wb')
        try:
            f.write(row.encode('utf-8'))
            if i % 1000 == 0:
                f.close()
                counter += 1
        except Exception as e:  #catch exceptions(for eg. empty rows)
            print(row, e) 
        i+=1
    # Return the last ID
    return counter