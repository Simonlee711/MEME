import numpy as np
from sagemaker import get_execution_role
import boto3
import pandas as pd
from io import StringIO  # Python 3.
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

from transformers import (
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json

import os
from tqdm import tqdm


def train_validate_test_split(df, train_percent=0.7, validate_percent=0.15, seed=7):
    np.random.seed(seed)  # set seed for reproducibility sake
    df = df.reset_index()
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    train = train.set_index("index")
    validate = validate.set_index("index")
    test = test.set_index("index")
    return train, validate, test


def train_validate_test_split(
    df, target_column, train_percent=0.7, validate_percent=0.15, seed=None
):
    np.random.seed(seed)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the dataframe
    targets = df[target_column].unique()
    train_indices, test_indices = train_test_split_by_stratified_indices(
        df, target_column, targets, train_percent
    )
    validate_percent_of_remaining = validate_percent / (1 - train_percent)
    validate_indices, test_indices = train_test_split_by_stratified_indices(
        df.loc[test_indices], target_column, targets, validate_percent_of_remaining
    )
    return df.loc[train_indices], df.loc[validate_indices], df.loc[test_indices]


def train_test_split_by_stratified_indices(df, target_column, targets, train_percent):
    train_indices = []
    test_indices = []
    for target in targets:
        target_indices = df[df[target_column] == target].index.to_list()
        np.random.shuffle(target_indices)
        train_size = int(train_percent * len(target_indices))
        train_indices.extend(target_indices[:train_size])
        test_indices.extend(target_indices[train_size:])
    return train_indices, test_indices
