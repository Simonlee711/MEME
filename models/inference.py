#%% Python
"""
python main.py \
    --model=./medbert \
    --weights=/opt/data/commonfilesharePHI/slee/GeneratEHR/pickle-models/generatEHR-disposition.pkl \
    --gpu=0
    --data=/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/text_repr.json \
    --num_workers=8 \
    --batch=64 \
    --experiment_name=JC_Multimodal_MedBert-UCLA-Inference \
    --log_dir=logs/ \
    --plot
"""
import logging
import argparse 
import os
import pickle
import json 
import traceback

import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt 

from fastprogress.fastprogress import master_bar, progress_bar

import torch
import torch.nn as nn
from datasets import Dataset, load_metric, DatasetDict
from transformers import DataCollatorWithPadding,  AutoTokenizer

from torch.utils.data import DataLoader

# %%
def parse_args():
    parser = argparse.ArgumentParser(
                        prog='Inference Program',
                        description='Runs inference on a dataset',
                        epilog='Additional Text at the bottom of help')

    parser.add_argument('-c', '--config', type=str, default='./config.json', help="path to the config")           # positional argument
    #     parser.add_argument('--model', type=str, default='./medbert', help="path to the model")           # positional argument
    #     parser.add_argument('--weights', type=str, help="path to the weights")           # positional argument
    #     parser.add_argument('--data', type=str, help="path to the data",
    #                         default='/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/text_repr.json')           # positional argument
    #     parser.add_argument('--gpu', type=int, help="GPU ID")           # positional argument
    #     parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    #     parser.add_argument('-b', '--batch', type=int, default=1, help="batch size")
    #     parser.add_argument('--experiment_name', default='experiment', help="Name of experiment")
    #     parser.add_argument('--log_dir', default='logs', help="Log directory")
    #     # TODO: If this is single mode, change the data loader...
    #     parser.add_argument('--mode', default='multimodal', help='Operating mode') 
    #     # parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args

def create_logger(log_dir, experiment_name):
    logging.basicConfig(
        filename = os.path.join(log_dir, f"{experiment_name}.log"),
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    return logger 

def load_data_UCLA(file_name):
    """assumption: data is in correct format"""
    with open(file_name) as f:
        json_content = json.load(f)
    df = pd.DataFrame(json_content).T
    # NOTE: THIS SHOULD CHANGE
    df["eddischarge"] = [
        0 if "h o m e" in s.lower() else 1 for s in df["eddischarge_category"]
    ]  # admitted = 1, Home = 0
    df["medrecon"] = "Previous medications were not recorded."

    df["triage"] = df["triage"].fillna("Triage measurements were not taken.")
    df["pyxis"] = df["pyxis"].fillna("The patient did not receive any medications.")
    df["vitals"] = df["vitals"].fillna("The patient had no vitals recorded")
    df["codes"] = df["codes"].fillna("The patient had no diagnostic codes recorded.")
    df = df.drop("admission", axis=1)
    df = df.drop("discharge", axis=1)
    # df = df.drop("eddischarge_category",axis=1)
    df["ID"] = (
        df.arrival.astype(str)
        .str.split()
        .str[1]
        .replace(",", " ", regex=True)
        .to_list()
    )
    df = df[
        [col for col in df.columns if col != "eddischarge"] + ["eddischarge"]
    ]  # rearrange column to the end
    return df

def load_data_MIMIC(file_name):
    """ assumption: data is in correct format """
    with open(file_name) as f:
        json_content = json.load(f)
    df = pd.DataFrame(json_content).T
    
#     df = df.iloc[:10000] # test
    # NOTE: THIS SHOULD CHANGE
    # df['eddischarge'] = [0 if 'a d m i t' in s.lower() else 1 for s in df['eddischarge_category']] # admitted = 1, Home = 0
    df['eddischarge'] = [0 if 'h o m e' in s.lower() else 1 for s in df['eddischarge_category']] # admitted = 1, Home = 0
    # df['eddischarge'] = [1 if 'admitted' in s.lower() else 0 for s in df['eddischarge']] # admitted = 1, Home = 0
    # df['medrecon'] = df['medrecon'].fillna("The patient was previously not taking any medications.")
    # since they're all missing...
    # df['medrecon'] = df['medrecon'].fillna("Previous medications were not recorded.")
    df['medrecon'] = "Previous medications were not recorded."

    df['triage'] = df['triage'].fillna("Triage measurements were not taken.")
    df['pyxis'] = df['pyxis'].fillna("The patient did not receive any medications.")
    df['vitals'] = df['vitals'].fillna("The patient had no vitals recorded")
    df['codes'] = df['codes'].fillna("The patient had no diagnostic codes recorded.")
    df = df.drop("admission",axis=1)
    df = df.drop("discharge",axis=1)
    # df = df.drop("eddischarge_category",axis=1)
    df['ID'] = df.arrival.astype(str).str.split().str[1].replace(",", " ", regex=True).to_list()
    df = df[[col for col in df.columns if col != 'eddischarge'] + ['eddischarge']] # rearrange column to the end
    return df 

def append_multi_bench_UCLA(df, file_name):
    # generate dictionaries from the icustays dataset to map to our dataset
    metainfo = pd.read_csv(file_name, sep='$')
    metainfo['hadm_id'] = metainfo['hadm_id'].astype(str)
    
    metainfo['further_discharge'] = (1 - (metainfo['DischargeDisposition'] == 'Home or Self Care').astype(int)).fillna(1) # assume home unless otherwise specified
    metainfo['mortality'] = metainfo['hospital_expire_flag'].fillna(0)
    metainfo['ICU'] = metainfo['ICU'].fillna(0)
    metainfo['labels'] = metainfo.apply(lambda row: [row['further_discharge'], row['mortality'], row['ICU']], axis=1)

    return df\
        .merge(metainfo.set_index('hadm_id')[['labels']], left_index=True, right_index=True, how='left')\

def append_multi_bench_MIMIC(df):
    # generate dictionaries from the icustays dataset to map to our dataset
    df["stay_id"] = df.index
    df["stay_id"] = df["stay_id"].astype("int64")
    df["ID"] = df["ID"].astype("int64")

    # add hadm_id to dataset
    edstays = pd.read_csv(
        "/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/mimic/mimic-iv-ed-2.2/mimic-iv-ed-2.2/ed/edstays.csv.gz"
    )
    edstays = edstays.fillna(0)
    edstays["hadm_id"] = edstays["hadm_id"].astype(int)
    edstays = edstays[["stay_id", "hadm_id"]]
    metainfo = pd.merge(df, edstays, on=["stay_id"])

    # generate dictionaries from the admissions dataset to map to our dataset
    admissions = pd.read_csv("./mimic-iv/admissions.csv.gz")
    discharge_dict = admissions.set_index("hadm_id")["discharge_location"].to_dict()
    death_dict = admissions.set_index("hadm_id")["hospital_expire_flag"].to_dict()

    # generate dictionaries from the icustays dataset to map to our dataset
    icustays = pd.read_csv("./mimic-iv/icustays.csv.gz")
    icustays["ICU"] = 1
    ICU_dict = icustays.set_index("hadm_id")["ICU"].to_dict()

    # map and fill in zeros
    metainfo["discharge_location"] = metainfo["hadm_id"].map(discharge_dict)
    metainfo["hospital_expire_flag"] = metainfo["hadm_id"].map(death_dict)
    l = metainfo.discharge_location.value_counts()
    metainfo["ICU"] = metainfo["hadm_id"].map(ICU_dict)
    metainfo["discharge_location"] = metainfo["discharge_location"].fillna("HOME")
    metainfo["discharge_location"] = (
        metainfo["discharge_location"].isin(["HOME"]).astype(int)
    )
    metainfo["discharge_location"] = (
        ~metainfo["discharge_location"].astype(bool)
    ).astype(
        int
    )  # Home - 0 Other - 1
    metainfo = metainfo.rename(columns={"hospital_expire_flag": "mortality"})
    metainfo = metainfo.rename(columns={"discharge_location": "further_discharge"})

    metainfo = metainfo.fillna(0.0)
    metainfo["further_discharge"] = metainfo["further_discharge"].astype(int)
    metainfo["mortality"] = metainfo["mortality"].astype(int)
    metainfo["ICU"] = metainfo["ICU"].astype(int)
    metainfo["labels"] = metainfo.apply(
        lambda row: [row["further_discharge"], row["mortality"], row["ICU"]], axis=1
    )
    metainfo = metainfo.drop("further_discharge", axis=1)
    metainfo = metainfo.drop("mortality", axis=1)
    metainfo = metainfo.drop("ICU", axis=1)
    metainfo = metainfo.drop("ID", axis=1)
    metainfo = metainfo.drop("hadm_id", axis=1)
    metainfo.index = metainfo.stay_id
    metainfo = metainfo.drop("stay_id", axis=1)
    metainfo.index.name = None
    metainfo = metainfo[metainfo["eddischarge"] != 0]
    return metainfo
    

def cut(df, set_type, label_col="eddischarge"):
    col_names = df.columns.drop(label_col)
    l = []
    for i in col_names:
        temp = df[[i, label_col]].reset_index()
        temp = temp.sort_values(by=['index']).reset_index() # we sort the patient ID numerically before dropping it to preserve order in encoding
        temp = temp.drop(columns=["index", "level_0"])
        temp = temp.rename(columns={i: "headline", label_col: "label"})
        l.append(temp)
        print("\""+i+ "\" Dataframe:", set_type, "set has been split")
    return l

# modified JNC 12/5/2023
class Tokenizer():
    def __init__(self, tokenizer_name_or_path='./medbert'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path) # "distilbert-base-uncased")

    def tokenize(self,examples):
        """Mapping function to tokenize the sentences passed with truncation"""
        return self.tokenizer(examples["headline"], truncation=True, padding="max_length",
                        max_length=512, return_special_tokens_mask=True)
                        
    def convert(self, l):
        """
        Run this method
        """
        arrival_hf=Dataset.from_pandas(l[0])
        triage_hf=Dataset.from_pandas(l[2]) # 1])
        medrecon_hf=Dataset.from_pandas(l[3])
        vitals_hf=Dataset.from_pandas(l[4])
        codes_hf=Dataset.from_pandas(l[5])
        pyxis_hf=Dataset.from_pandas(l[6])

        arrival = arrival_hf.map(self.tokenize, batched=True, num_proc=NUM_WORKERS)
        triage = triage_hf.map(self.tokenize, batched=True, num_proc=NUM_WORKERS)
        medrecon = medrecon_hf.map(self.tokenize, batched=True, num_proc=NUM_WORKERS)
        vitals = vitals_hf.map(self.tokenize, batched=True, num_proc=NUM_WORKERS)
        codes = codes_hf.map(self.tokenize, batched=True, num_proc=NUM_WORKERS)
        pyxis = pyxis_hf.map(self.tokenize, batched=True, num_proc=NUM_WORKERS)
        
        print(arrival)

        arrival.set_format('torch', columns=["input_ids", "attention_mask", "label"] )
        triage.set_format('torch', columns=["input_ids", "attention_mask", "label"] )
        medrecon.set_format('torch', columns=["input_ids", "attention_mask", "label"] )
        vitals.set_format('torch', columns=["input_ids", "attention_mask", "label"] )
        codes.set_format('torch', columns=["input_ids", "attention_mask", "label"] )
        pyxis.set_format('torch', columns=["input_ids", "attention_mask", "label"] )

        return arrival, triage, medrecon, vitals, codes, pyxis

def single_mode_dataloader(tokens, tokenizer, batch, num_workers):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataset_cc = DatasetDict({
        'test': tokens,
    })
    
    dataloader = DataLoader(
        dataset_cc['test'], 
        batch_size = batch,
        # num_workers=num_workers,
        collate_fn=data_collator)    
    print('dataloader created')
    return dataloader

    
def multimodal_dataloader(arrival,triage,medrecon,vitals,codes,pyxis,tokenizer,batch=1, num_workers=0):
    all_input_ids = np.stack([
        arrival['input_ids'].numpy(),
        triage['input_ids'].numpy(),
        medrecon['input_ids'].numpy(),
        vitals['input_ids'].numpy(),
        codes['input_ids'].numpy(),
        pyxis['input_ids'].numpy()
    ], axis=1)

    all_attn_weights = np.stack([
        arrival['attention_mask'].numpy(),
        triage['attention_mask'].numpy(),
        medrecon['attention_mask'].numpy(),
        vitals['attention_mask'].numpy(),
        codes['attention_mask'].numpy(),
        pyxis['attention_mask'].numpy()
    ], axis=1)

    all_labels = arrival['label'].numpy()

    # all_tokens.shape, all_attn_weights.shape, all_labels.shape


    multimodal_dataset = Dataset.from_dict({
        'input_ids': [list(v) for v in all_input_ids],
        'attention_mask': [list(v) for v in all_attn_weights],
        'label': all_labels,
    })

    multimodal_dataset = multimodal_dataset.with_format('torch')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    multimodal_dataloader = DataLoader(
        multimodal_dataset, 
        batch_size = batch,
        # num_workers=num_workers, # seems to blow up 
        collate_fn=data_collator)  
    print("dataloader created")
    return multimodal_dataloader

def load_pkl_model(file, device):
    # file = "../models/models-bin/arrival.pkl"
    print(f"Loading... {file}")
    with open(file, 'rb') as f:
        model_task_specific = pickle.load(f)        
        model_task_specific.to(device)
    return model_task_specific

def load_torch_model(file, device, mode, task):
    print(f"Loading model from {file}")
    # Create an instance of the ED_classifier
    if mode == "multimodal":
        if task =="multitask":
            model_task_specific = ED_classifier(checkpoint="./medbert", num_labels=3, input_dim=768, modalities=6, freeze=True)
        if task == "eddispo":
            model_task_specific = ED_classifier(checkpoint="./medbert", num_labels=2, input_dim=768, modalities=6, freeze=True)
    if mode == "single":
        if task == "multitask":            
            model_task_specific = SingleModPredictor(checkpoint="./medbert", num_labels=3, freeze=True)
        if task == "eddispo":            
            model_task_specific = SingleModPredictor(checkpoint="./medbert", num_labels=2, freeze=True)
    # Load the state dictionary into the model
    model_task_specific.load_state_dict(torch.load(file, map_location=device))
    model_task_specific.to(device)
    return model_task_specific

def inference_multitask(model_name, model, test_dataloader, device, metric=None, mode='multimodal', plot=False):

    if metric is None: 
        metric = load_metric("f1")
    probs_list = []
    label_list = []
    logits_list = []

    for batch in progress_bar(test_dataloader):
        if mode == 'multimodal':
            input_ids = [b.to(device) for b in torch.unbind(batch['input_ids'], dim=1)]
            attention_masks = [b.to(device) for b in torch.unbind(batch['attention_mask'], dim=1)]
        else:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad(): # a context manager that disables gradient calculation during model inference
            outputs = model(input_ids, attention_masks, labels)

        sigmoid_output = torch.sigmoid(outputs.logits)
        predictions = (sigmoid_output > 0.5).to(torch.int)
        metric.add_batch(predictions = predictions.view(-1), references = batch['labels'].view(-1) )
        
        logits_list.append(outputs.logits.cpu().detach().numpy())
        probs_list.append(sigmoid_output.cpu().detach().numpy())
        label_list.append(batch['labels'].cpu().detach().numpy())      
        

    print(metric.compute()) 
    # stack vertically!
    # yt = np.vstack(label_list) 
    # yl = np.vstack(logits_list)
    # yp = np.vstack(probs_list)

    result_df = pd.concat([
        pd.DataFrame(np.vstack(label_list), columns=['ytrue_home', 'ytrue_mortality', 'ytrue_icu']),
        pd.DataFrame(np.vstack(logits_list), columns=['yraw_home', 'yraw_mortality', 'yraw_icu']),
        pd.DataFrame(np.vstack(probs_list), columns=['yprob_home', 'yprob_mortality', 'yprob_icu']),
        
    ], axis=1)


    return result_df

def inference_eddispo(model_name, model, test_dataloader, device, metric=None, mode='multimodal', plot=False):

    if metric is None: 
        metric = load_metric("f1")
    probs_list = []
    label_list = []
    logits_list = []

    model.eval()

    for batch in progress_bar(test_dataloader):
        # batch = { k: v for k, v in batch.items() }
        if mode == 'multimodal':
            input_ids = [b.to(device) for b in torch.unbind(batch['input_ids'], dim=1)]
            attention_masks = [b.to(device) for b in torch.unbind(batch['attention_mask'], dim=1)]
        else:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad(): # a context manager that disables gradient calculation during model inference
            outputs = model(input_ids, attention_masks, labels)

        logits = outputs.logits # calculates the probabilities between the labels
        probs_list.append(torch.sigmoid(logits[:, 1]).cpu().detach().numpy())
        logits_list.append(logits[:, 1].cpu().detach().numpy())
        label_list.append(batch['labels'].cpu().detach().numpy())
        predictions = torch.argmax(logits, dim = -1 ) # takes the label closest to 1
        metric.add_batch(predictions = predictions, references = batch['labels'] )
        
    print(metric.compute()) 
    yt = np.hstack(label_list) 
    yl = np.hstack(logits_list)
    yp = np.hstack(probs_list)

    result_df = pd.DataFrame({"y_true": yt, "y_prob": yp, "y_raw": yl})

    return result_df

# %%
if __name__ == '__main__':    
    args=parse_args()
    with open(args.config) as f:
        config = json.load(f)
    
    # %%
    LOG_DIR = config.get('log_dir', 'logs')
    EXPERIMENT_NAME = config.get('experiment_name', 'experiment')
    BATCH = config.get('batch', 1)
    NUM_WORKERS = config.get('num_workers', 0) 
    DATA = config.get('data', '/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/text_repr.json')
    MODEL = config.get('model', './medbert')
    MODE = config.get("mode", 'multimodal')
    WEIGHTS = config.get('weights')
    GPU = config.get('gpu')
    TASK = config.get("task", "eddispo")
    FOLDER = config.get("folder" "./")
    DATASET = config.get("dataset", "UCLA")
    assert WEIGHTS is not None, "Weights directory missing"
    
    logger = create_logger(LOG_DIR, EXPERIMENT_NAME)

    DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU is not None else "cpu")
    # %%
    print(f'LOGGING TO {LOG_DIR}')
    logger.info("--------------------------------")
    logger.info(f'TRAINING ON {DEVICE}')
    logger.info(f'BATCH SIZE: {BATCH} | NUM_WORKERS: {NUM_WORKERS}')
    logger.info(f'DATA DIR: {DATA}')
    logger.info(f'MODEL AND TOKENIZER: {MODEL}')
    logger.info(f'MODEL AND WEIGHTS: {WEIGHTS}')
    logger.info(f'OPERATING MODE: {MODE}')
    logger.info(f'EXPERIMENT NAME: {EXPERIMENT_NAME}')
    logger.info(f'TASK: {TASK}')
    logger.info(f'FOLDER: {FOLDER}')
    logger.info(f'DATASET: {DATASET}')
    # %%
    try:
        if TASK == "eddispo":
            from model import *
        if TASK == "multitask":
            from model_multitask import *
        # %%
        if DATASET == "UCLA":
            logger.info('loading data...')
            df = load_data_UCLA(DATA)
            label_col = "eddischarge"

            # %%

            if TASK == "multitask":
                logger.info("Appending multitask data...")
                df = append_multi_bench_UCLA(df, "/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/er-pull.rpt")
                df = df[df["eddischarge"] != 0]
                df = df.drop("eddischarge",axis=1)
                label_col = "labels"
        if DATASET == "MIMIC":
            logger.info('loading data...')
            df = load_data_MIMIC(DATA)
            label_col = "eddischarge"

            # %%

            if TASK == "multitask":
                logger.info("Appending multitask data...")
                df = append_multi_bench_MIMIC(df)
                df = df[df["eddischarge"] != 0]
                df = df.drop("eddischarge",axis=1)
                label_col = "labels"
            
        # %%

        logger.info('splitting data...')
        l_all = cut(df, "test", label_col=label_col)
        # bucket_name = '/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/text_repr.json'

        # %%
        # calls methods and tokenizes text
        logger.info('loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(MODEL) # "distilbert-base-uncased")
        processor = Tokenizer()
        
        logger.info('tokenizing...')
        arrival_test_tokens, triage_test_tokens, medrecon_test_tokens, vitals_test_tokens, codes_test_tokens, pyxis_test_tokens, = processor.convert(l_all)
        data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer)

        # %%
        # train_multimodal_dataloader = multimodal_dataloader(arrival_train_tokens, triage_train_tokens, medrecon_train_tokens, vitals_train_tokens, codes_train_tokens, pyxis_train_tokens,tokenizer,4)
        # validate_multimodal_dataloader = multimodal_dataloader(arrival_val_tokens, triage_val_tokens, medrecon_val_tokens, vitals_val_tokens, codes_val_tokens, pyxis_val_tokens,tokenizer,4)
        logger.info('creating dataloader...')
        if MODE == 'multimodal':
            test_multimodal_dataloader = multimodal_dataloader(
                arrival_test_tokens, 
                triage_test_tokens, 
                medrecon_test_tokens, 
                vitals_test_tokens,
                codes_test_tokens, 
                pyxis_test_tokens,
                processor.tokenizer,
                batch=BATCH,
                num_workers=NUM_WORKERS
            )
            # %%
            logger.info('loading model and weights...')
            model = load_torch_model(WEIGHTS, DEVICE, MODE, TASK)

            logger.info('inference...')
            if TASK == 'eddispo':
                result_df = inference_eddispo(
                    model_name=EXPERIMENT_NAME, 
                    model=model,
                    test_dataloader=test_multimodal_dataloader,
                    device=DEVICE,
                    metric=None, 
                    mode=MODE,
                    # plot=args.plot
                )
            if TASK == 'multitask':
                result_df = inference_multitask(
                    model_name=EXPERIMENT_NAME, 
                    model=model,
                    test_dataloader=test_multimodal_dataloader,
                    device=DEVICE,
                    metric=None, 
                    mode=MODE,
                    # plot=args.plot
                )
            # %%  
            result_df.set_index(df.index)
            result_df.to_csv(f'{EXPERIMENT_NAME}.csv')
            logger.info('done.')
        # %%
        if MODE == 'single':
            tokens_mapped = {
                'arrival': arrival_test_tokens
                , 'triage': triage_test_tokens
                , 'vitals': vitals_test_tokens
                , 'codes': codes_test_tokens
                , 'pyxis': pyxis_test_tokens
            }
            for mode, tokens in tokens_mapped.items():
                logging.info(f'Running {mode}')
                exp_name = f'{mode}-{EXPERIMENT_NAME}'
                test_single_dataloader = single_mode_dataloader(
                    tokens=tokens, 
                    tokenizer=processor.tokenizer, 
                    batch=BATCH, 
                    num_workers=NUM_WORKERS)
        
                if TASK == 'eddispo':
                    logger.info('loading model and weights...')
                    m_weights = os.path.join(WEIGHTS, f'{mode}-disposition.pth')
                    model = load_torch_model(m_weights, DEVICE, MODE, TASK)

                    logger.info('inference...')
                    result_df = inference_eddispo(
                        model_name=EXPERIMENT_NAME, 
                        model=model,
                        test_dataloader=test_single_dataloader,
                        device=DEVICE,
                        metric=None, 
                        mode=MODE,
                        # plot=args.plot
                    )
                if TASK == 'multitask':
                    logger.info('loading model and weights...')
                    m_weights = os.path.join(WEIGHTS, f'{mode}-multitask.pth')
                    model = load_torch_model(m_weights, DEVICE, MODE, TASK)

                    logger.info('inference...')
                    result_df = inference_multitask(
                        model_name=EXPERIMENT_NAME, 
                        model=model,
                        test_dataloader=test_single_dataloader,
                        device=DEVICE,
                        metric=None, 
                        mode=MODE,
                        # plot=args.plot
                    )        
                result_df.set_index(df.index)
                result_df.to_csv(f'{exp_name}.csv')
                logger.info('done.')
    # %%
    except Exception as e:
        logger.error(traceback.format_exc())

