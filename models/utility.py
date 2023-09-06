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

def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=7):
    np.random.seed(seed) # set seed for reproducibility sake
    df = df.reset_index()
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    train = train.set_index('index')
    validate = validate.set_index('index')
    test = test.set_index('index')
    return train, validate, test