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

import os
from tqdm import tqdm

class CustomDataset(Dataset):
    """
    Custom Dataset class to load into transformers for pretraining 
    """
    def __init__(self, df, tokenizer):
        # or use the RobertaTokenizer from `transformers` directly.
        self.examples = []
        # For every value in the dataframe 
        for example in tqdm(df.values):
            # 
            x=tokenizer.encode_plus(example,max_length = 512, truncation=True, padding=True)
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])
