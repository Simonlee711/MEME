!pip install transformers datasets[s3] sagemaker --upgrade
!pip install scikit-learn
!pip install accelerate==0.20.3

import numpy as np
from sagemaker import get_execution_role
import boto3
import pandas as pd
from io import StringIO # Python 3.
from datasets import load_dataset,Dataset,DatasetDict,concatenate_datasets
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import json
import pickle
from transformers import AdamW, get_scheduler
from datasets import load_metric

from singlemodalmodel import SingleModPredictor
from multimodalmodel import EDDispositionFineTuneModel
from tokenizer import Tokenizer

def preprocess(df):
    """
    Preprocesses the EHR Dataframe.

    This function performs the following operations on the DataFrame:
    1. Converts 'eddischarge' to binary labels: 'admitted' is labeled as 1, 'Home' as 0.
    2. Fills missing values in 'medrecon', 'pyxis', and 'vitals' with appropriate messages.
    3. Fills missing values in 'codes' with an appropriate message.
    4. Drops 'admission', 'discharge', and 'eddischarge_category' columns.
    5. Extracts the ID from the 'arrival' column and assigns it to the 'ID' column.
    6. Rearranges the columns to have 'eddischarge' at the end.

    Args:
    df (DataFrame): The DataFrame to be preprocessed.

    Returns:
    DataFrame: The preprocessed DataFrame.

    """
    print("Preprocessing our EHR Dataframe")
    df['eddischarge'] = [1 if 'admitted' in s.lower() else 0 for s in df['eddischarge']] # admitted = 1, Home = 0
    df['medrecon'] = df['medrecon'].fillna("The patient was previously not taking any medications.")
    df['pyxis'] = df['pyxis'].fillna("The patient did not receive any medications.")
    df['vitals'] = df['vitals'].fillna("The patient had no vitals recorded")
    df['codes'] = df['codes'].fillna("The patient received no diagnostic codes")
    df = df.drop("admission", axis=1)
    df = df.drop("discharge", axis=1)
    df = df.drop("eddischarge_category", axis=1)
    df['ID'] = df.arrival.astype(str).str.split().str[1].replace(",", " ", regex=True).to_list()
    df = df[[col for col in df.columns if col != 'eddischarge'] + ['eddischarge']] 
    print("Finished preprocessing our EHR Dataframe")
    return df

def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=None):
    """
    Splits the input DataFrame into training, validation, and test sets.

    This function shuffles the DataFrame and splits it into three portions based on the percentages provided.
    The training set gets the first portion, the validation set gets the second, and the test set gets the rest.

    Args:
    df (DataFrame): The DataFrame to be split.
    train_percent (float, optional): The percentage of the DataFrame to be allocated for the training set.
    validate_percent (float, optional): The percentage of the DataFrame to be allocated for the validation set.
    seed (int, optional): The seed value for the random number generator.

    Returns:
    tuple: A tuple containing the training, validation, and test DataFrames.

    """
    np.random.seed(seed)
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

def write_patient(train, validate, test, inference=False):
    """
    Writes the patient lists to pickle files and reads them back if 'inference' is set to True.

    Args:
    train (DataFrame): The DataFrame containing the training patient cohort data.
    validate (DataFrame): The DataFrame containing the validation patient cohort data.
    test (DataFrame): The DataFrame containing the test patient cohort data.
    inference (bool, optional): A flag that specifies whether to read the patient lists or write them to pickle files.

    Returns:
    tuple: A tuple containing the patient lists for the training, validation, and test sets.

    """
    if not inference:
        train_list = train["ID"].to_list()
        validation_list = validate["ID"].to_list()
        test_list = test["ID"].to_list()
        print(len(train_list), len(validation_list), len(test_list))
        file_path = "./models/data/train_patients.pkl"
        file_path2 = "./models/data/validation_patients.pkl"
        file_path3 = "./models/data/test_patients.pkl"
        # Write the list to the pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(train_list, file)
        with open(file_path2, 'wb') as file:
            pickle.dump(validation_list, file)
        with open(file_path3, 'wb') as file:
            pickle.dump(test_list, file)
    else:
        file_path = "./models/data/train_patients.pkl"
        file_path2 = "./models/data/validation_patients.pkl"
        file_path3 = "./models/data/test_patients.pkl"
        # Read the list from the pickle file
        with open(file_path, 'rb') as file:
            train_list = pickle.load(file)
        with open(file_path2, 'rb') as file:
            validation_list = pickle.load(file)
        with open(file_path3, 'rb') as file:
            test_list = pickle.load(file)
        print(len(train_list), len(validation_list), len(test_list))
        print(type(train_list), type(validation_list), type(test_list))
        return train_list, validation_list, test_list
    
def cut(df, set_type):
    """
    Cuts the DataFrame into separate DataFrames for different modality types so it can be tokenized.

    Args:
    df (DataFrame): The DataFrame to be split.
    set_type (str): The type of the set, e.g., 'train', 'validation', 'test'.

    Returns:
    list: A list of DataFrames, each representing a different subset of the input DataFrame.

    """
    col_names = df.columns.drop("eddischarge")
    l = []
    for i in col_names:
        temp = df[[i, 'eddischarge']].reset_index()
        temp = temp.sort_values(by=['index']).reset_index()  # We sort the patient ID numerically before dropping it to preserve order in encoding
        temp = temp.drop(columns=["index", "level_0"])
        temp = temp.rename(columns={i: "headline", "eddischarge": "label"})
        l.append(temp)
        print("\"" + i + "\" Dataframe:", set_type, "set has been split")
    return l

def model_param_load():
    """
    Loads model parameters and initializes an optimizer and learning rate scheduler.

    Returns:
    tuple: Tuple containing optimizer, learning rate scheduler, and the metric.

    """
    # optimizer = AdamW(model_task_specific.parameters(), lr = 5e-5 )
    optimizer = AdamW(full_model.parameters(), lr=5e-5)

    num_epoch = 1
    num_training_steps = num_epoch * len(triage_dataset_cc['train']["input_ids"])
    print(len(triage_dataset_cc['train']["input_ids"]))

    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    metric = load_metric("f1")

    return optimizer, lr_scheduler, metric

def train_single(model_task_specific, train_dataloader, eval_dataloader, optimizer, lr_scheduler, metric):
    """
    Trains a single modality model and evaluates its performance.

    Args:
    model_task_specific (torch.nn.Module): The PyTorch model to be trained and evaluated.
    train_dataloader (torch.utils.data.DataLoader): The data loader for the training data.
    eval_dataloader (torch.utils.data.DataLoader): The data loader for the evaluation data.
    optimizer: The optimizer used for updating weights and biases based on gradients.
    lr_scheduler: The learning rate scheduler used for updating the learning rate during training.
    metric: The metric used for evaluating the model's performance.

    Returns:
    None

    """
    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(num_epoch * len(eval_dataloader)))

    for epoch in range(num_epoch):
        model_task_specific.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_task_specific(**batch)
            loss = outputs.loss
            loss.backward()  # computes gradients

            optimizer.step()  # updates the weights and biases based on these gradients
            lr_scheduler.step()  # updates the weights and biases based on these gradients
            optimizer.zero_grad()  # used to clear the gradients of all parameters in a model
            # progress_bar_train.update(1)
        print("epoch training", str(epoch), "done")
        print("loss:", str(loss))

        # run on validation set
        model_task_specific.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():  # a context manager that disables gradient calculation during model inference
                outputs = model_task_specific(**batch)

            logits = outputs.logits  # calculates the probabilities between the labels
            predictions = torch.argmax(logits, dim=-1)  # takes the label closest to 1
            metric.add_batch(predictions=predictions, references=batch['labels'])
            # progress_bar_eval.update(1)

        print(metric.compute())

def test_single(model_task_specific, test_dataloader, metric):
    """
    Tests a single modality model and evaluates its performance on the test dataset.

    Args:
    model_task_specific (torch.nn.Module): The PyTorch model to be tested.
    test_dataloader (torch.utils.data.DataLoader): The data loader for the test data.
    metric: The metric used for evaluating the model's performance.

    Returns:
    tuple: A tuple containing three lists - logit_list, label_list, and probs_list.
    logit_list (list): A list of logits calculated from the test data.
    label_list (list): A list of labels from the test data.
    probs_list (list): A list of probabilities calculated from the test data.

    """
    probs_list = []
    label_list = []
    logits_list = []

    test_dataloader = DataLoader(
        pyxis_dataset_cc['test'], batch_size=4, collate_fn=data_collator
    )

    metric = load_metric("f1", "precision")

    for i, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():  # a context manager that disables gradient calculation during model inference
            outputs = model_task_specific(**batch)

        logits = outputs.logits  # calculates the probabilities between the labels
        probs_list.append(torch.sigmoid(logits[:, 1]).cpu().detach().numpy())
        logits_list.append(logits[:, 1].cpu().detach().numpy())
        label_list.append(batch['labels'].cpu().detach().numpy())
        predictions = torch.argmax(logits, dim=-1)  # takes the label closest to 1
        metric.add_batch(predictions=predictions, references=batch['labels'])

    print(metric.compute())
    return logit_list, label_list, probs_list
        
def train_multi(full_model, train_dataloader_concat, eval_dataloader_concat, optimizer, lr_scheduler, metric):
    """
    Trains a multi-modal model and evaluates its performance on the concatenated train and evaluation dataloaders.

    Args:
    full_model (torch.nn.Module): The PyTorch multi-modal model to be trained.
    train_dataloader_concat (list): List of concatenated training dataloaders for different modalities.
    eval_dataloader_concat (list): List of concatenated evaluation dataloaders for different modalities.
    optimizer: The optimizer used for training the model.
    lr_scheduler: The scheduler used for adjusting the learning rate during training.
    metric: The metric used for evaluating the model's performance.

    """
    num_epoch = 1
    BATCH = 4
    from tqdm.auto import tqdm
    progress_bar_train = tqdm(range(num_epoch * len(triage_dataset_cc['train']["input_ids"]) // BATCH))
    progress_bar_eval = tqdm(range(num_epoch * len(triage_dataset_cc['valid']["input_ids"]) // BATCH))

    for epoch in range(num_epoch):
        full_model.train()
        print(f"Epoch {epoch}...")
        random_idx = np.random.permutation(np.arange(len(train_dataloader_concat[0]['input_ids'])))
        for step, idx in enumerate(range(0, len(random_idx), BATCH)):
            iter_rand_idx = random_idx[idx:idx+BATCH]
            input_ids, attention_mask = [], []
            for modality in train_dataloader_concat:
                input_ids.append(modality['input_ids'][iter_rand_idx].to(device))
                attention_mask.append(modality['attention_mask'][iter_rand_idx].to(device))
            label = modality['label'][iter_rand_idx].to(device)
            outputs = full_model(input_ids, attention_mask, label)

            # updates weights accordingly
            loss = outputs.loss
            loss.backward() # computes gradients

            optimizer.step() # updates the weights and biases based on these gradients
            lr_scheduler.step() # updates the weights and biases based on these gradients
            optimizer.zero_grad() # used to clear the gradients of all parameters in a model
            progress_bar_train.update(1)

        # run on validation set
        print("Validation")
        full_model.eval()
        for step, idx in enumerate(range(0, len(valid_dataloader_concat[0]['input_ids']), BATCH)):
            input_ids, attention_mask = [], []
            for modality in valid_dataloader_concat:
                input_ids.append(modality['input_ids'][idx:idx+BATCH].to(device))
                attention_mask.append(modality['attention_mask'][idx:idx+BATCH].to(device))
            label = modality['label'][idx:idx+BATCH].to(device)
            with torch.no_grad():
                outputs = full_model(input_ids, attention_mask, label)
            logits = outputs.logits # calculates the probabilities between the labels
            predictions = torch.argmax(logits, dim=-1) # takes the label closest to 1
            metric.add_batch(predictions=predictions, references=label)
            loss = outputs.loss
            progress_bar_eval.update(1)

        print(metric.compute())

def test_multi(full_model, test_dataloader_concat, metric):
    """
    Tests the multi-modal model and evaluates its performance on the concatenated test dataloaders.

    Args:
    full_model (torch.nn.Module): The PyTorch multi-modal model to be tested.
    test_dataloader_concat (list): List of concatenated test dataloaders for different modalities.
    metric: The metric used for evaluating the model's performance.

    Returns:
    logit_list (list): List of logits computed during the test.
    label_list (list): List of labels used during the test.
    probs_list (list): List of probabilities computed during the test.

    """
    logit_list = []
    label_list = []
    probs_list = []

    full_model.eval()
    for step, idx in tqdm(enumerate(range(0, len(test_dataloader_concat[0]['input_ids']), BATCH))):
        input_ids, attention_mask = [], []
        for modality in test_dataloader_concat:
            input_ids.append(modality['input_ids'][idx:idx+BATCH].to(device))
            attention_mask.append(modality['attention_mask'][idx:idx+BATCH].to(device))
        label = modality['label'][idx:idx+BATCH].to(device)
        with torch.no_grad():
            outputs = full_model(input_ids, attention_mask, label)
        logits = outputs.logits  # calculates the probabilities between the labels
        predictions = torch.argmax(logits, dim=-1)  # takes the label closest to 1
        loss = outputs.loss
        logits = outputs.logits  # calculates the probabilities between the labels
        logit_list.append(logits[:, 1].cpu().detach().numpy())
        label_list.append(label.cpu().detach().numpy())
        probs_list.append(torch.sigmoid(logits[:, 1]).cpu().detach().numpy())
        predictions = torch.argmax(logits, dim=-1)  # takes the label closest to 1
        metric.add_batch(predictions=predictions, references=label)

    print(metric.compute())
    return logit_list, label_list, probs_list


if __name__ == '__main__': 
    ###### Loading Data ######
    # Load data from S3 bucket
    print("Loading mimic-iv-ed-2.2/text_repr.json")
    bucket_name = 'chianglab-dataderivatives'
    file_path = "mimic-iv-ed-2.2/text_repr.json"

    print("Finished Loading mimic-iv-ed-2.2/text_repr.json")
    s3 = boto3.resource('s3')
    content_object = s3.Object(bucket_name, file_path)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    df = pd.DataFrame(json_content).T
    
    ###### Preprocessing Dataframe ######
    # Preprocess the data
    df = preprocess(df)
    
    ###### Generate Train, Validation, Test Splits ######
    # Generate train, validation, and test splits
    # Keeping it consistent with our pretraining and finetuning efforts so we don't want repeat patients
    t, val, t2 = train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=7)
    remain = pd.concat([val, t2])

    # #resplit the our testing dataframe into an additional train and test split for fine tuning 
    train, validate, test =  train_validate_test_split(remain, seed=7)
    print("70% Train:",len(train), "\n30% Test:",len(validate+test))
    
    ###### Patient Writeout. ######
    # Write out patients you are working with. Sanity checker
    inference = True
    train_list, validate_list, test_list = write_patient(train, validate, test)
    
    ###### Pair Modality with Labels so it can be tokenized ######
    # Pair modalities with labels for tokenization
    print("################################################")
    l1 = cut(train, "train")
    print("################################################")
    l2 = cut(validate, "validation")
    print("################################################")
    l3 = cut (test, "test")
    print("################################################")

    ###### Tokenize the Data ######
    # Tokenize the data
    processor = Tokenizer()
    arrival_train_tokens, triage_train_tokens, medrecon_train_tokens, vitals_train_tokens, codes_train_tokens, pyxis_train_tokens, = processor.convert(l1)
    arrival_val_tokens, triage_val_tokens, medrecon_val_tokens, vitals_val_tokens, codes_val_tokens, pyxis_val_tokens, = processor.convert(l2)
    arrival_test_tokens, triage_test_tokens, medrecon_test_tokens, vitals_test_tokens, codes_test_tokens, pyxis_test_tokens, = processor.convert(l3)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    ###### Merge tokens into dataset object ######
    # Merge tokens into dataset objects
    arrival_dataset_cc = DatasetDict({
        'train': arrival_train_tokens,
        'test': arrival_test_tokens,
        'valid': arrival_val_tokens})

    triage_dataset_cc = DatasetDict({
        'train': triage_train_tokens,
        'test': triage_test_tokens,
        'valid': triage_val_tokens})

    medrecon_dataset_cc = DatasetDict({
        'train': medrecon_train_tokens,
        'test': medrecon_test_tokens,
        'valid': medrecon_val_tokens})

    vitals_dataset_cc = DatasetDict({
        'train': vitals_train_tokens,
        'test': vitals_test_tokens,
        'valid': vitals_val_tokens})

    codes_dataset_cc = DatasetDict({
        'train': codes_train_tokens,
        'test': codes_test_tokens,
        'valid': codes_val_tokens})

    pyxis_dataset_cc = DatasetDict({
        'train': pyxis_train_tokens,
        'test': pyxis_test_tokens,
        'valid': pyxis_val_tokens})
    
    ###### include model ######
    # Include the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_model = EDDispositionFineTuneModel(checkpoint=model, num_labels=2, input_dim=768, modalities=6).to(device)
    model_task_specific = SingleModPredictor(checkpoint=model, num_labels=2).to(device)
    print(device)
    
    ###### Load Model parameters ######
    # Load model parameters
    optimizer, lr_scheduler, metric = model_param_load()
    
    ###### Data Loader ######
    # Create data loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # change to whatever modality here
    train_dataloader = DataLoader(
        medrecon_dataset_cc['train'], shuffle = True, batch_size = 1, collate_fn = data_collator
    )

    eval_dataloader = DataLoader(
        medrecon_dataset_cc['valid'], shuffle = True, collate_fn = data_collator
    )
    test_dataloader = DataLoader(
    medrecon_dataset_cc['test'], batch_size = 4, collate_fn = data_collator
    )    
    ###### Run one single modality model ######
    # Run the single modality model
    train_single(model_task_specific, train_dataloader, eval_dataloader, optimizer, lr_scheduler, metric)
    logit_list, label_list, probs_list = test_single(model_task_specific, test_dataloader, metric)
    # Save predictions 
    yt = np.hstack(label_list) 
    yl = np.hstack(logit_list)
    yp = np.hstack(probs_list)
    result_df = pd.DataFrame({"y_true": yt, "y_prob": yp, "y_raw": yl})
    result_df.to_csv('medrecon-only-model.csv')
    
    
    ###### Multimodality ######
    # Train and test the multimodal model
    train_dataloader_concat = [triage_dataset_cc["train"], arrival_dataset_cc["train"],medrecon_dataset_cc["train"],vitals_dataset_cc["train"],codes_dataset_cc["train"],pyxis_dataset_cc["train"]]
    valid_dataloader_concat = [triage_dataset_cc["valid"], arrival_dataset_cc["valid"],medrecon_dataset_cc["valid"],vitals_dataset_cc["valid"],codes_dataset_cc["valid"],pyxis_dataset_cc["valid"]]
    test_dataloader_concat = [triage_dataset_cc["test"], arrival_dataset_cc["test"],medrecon_dataset_cc["test"],vitals_dataset_cc["test"],codes_dataset_cc["test"],pyxis_dataset_cc["test"]]
    
    # Train and test
    train_multi(full_model, train_dataloader_concat, eval_dataloader_concat, optimizer, lr_scheduler, metric)
    logit_list, label_list, probs_list = test_multi(full_model, test_dataloader_concat, metric)
    # Save predictions 
    yt = np.hstack(label_list) 
    yl = np.hstack(logit_list)
    yp = np.hstack(probs_list)
    result_df = pd.DataFrame({"y_true": yt, "y_prob": yp, "y_raw": yl})
    result_df.to_csv('Finetune-multimodal.csv')
    