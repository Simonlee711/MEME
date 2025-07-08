![GitHub](https://img.shields.io/github/license/Simonlee711/MEME) [![arXiv](https://img.shields.io/badge/npj-red.svg)](https://www.nature.com/articles/s41746-025-01777-x) [![PapersWithCode](https://img.shields.io/badge/PapersWithCode-Multimodal%20Clinical%20Pseudo%20Notes-lightblue.svg)](https://paperswithcode.com/paper/multimodal-clinical-pseudo-notes-for) [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-MEME-yellow.svg)](https://huggingface.co/Simonlee711/MEME) [![Hugging Face Model](https://img.shields.io/badge/DeepWiki-blue.svg)](https://deepwiki.com/Simonlee711/MEME/1-meme-system-overview)

# Clinical decision support using pseudo-notes from multiple streams of EHR data

# Abstract

Electronic health records (EHR) contain data from disparate sources, spanning various biological and temporal scales. In this work, we introduce the Multiple Embedding Model for EHR (MEME), a deep learning framework for clinical decision support that operates over heterogeneous EHR. MEME first converts tabular EHR into “pseudo-notes”, reducing the need for concept harmonization across EHR systems and allowing the use of any state-of-the-art, open source language foundation models. The model separately embeds EHR domains, then uses a self-attention mechanism to learn the contextual importance of these multiple embeddings. In a study of 400,019 emergency department visits, MEME successfully predicted emergency department disposition, discharge location, intensive care requirement, and mortality. It outperformed traditional machine learning models (Logistic Regression, Random Forest, XGBoost, MLP), EHR foundation models (EHR-shot, MC-BEC, MSEM), and GPT-4 prompting strategies. Due to text serialization, MEME also exhibited strong few-shot learning performance in an external, unstandardized EHR database.

# Motivation

Electronic Health Records are heterogenous containing a mixture of numerical, categorical, free text data. However traditional machine learning models struggle to learn representations of categorical data due to traditional one hot encoding schemes that result in sparse matrices. Therefore in this work we present **pseudo-notes**, which is a data transformation from EHR tabular data to text that allows us to leverage recent Large Language Models that have better understanding of English and context. These in turn generate better representations of EHR data as demonstrated by this research which benchmarks text vs tabular input as well as multimodal vs singular modality input. 

**TLDR** Multiple Embedding Model for EHR (MEME) outperforms all benchmark models on various tasks recorded in the Emergency Department. A further description is referenced in the paper.

### Pseudo-notes

![Model Image](https://github.com/Simonlee711/MEME/blob/main/img/model.png "Model Architecture")

# Data
The [MIMIC-IV-ED dataset](https://physionet.org/content/mimic-iv-ed/2.2/), part of the extensive MIMIC-IV collection, concentrates on emergency department records from a major hospital. It anonymizes and details patient demographics, triage, vitals, tests, medications, and outcomes, aiding research in emergency care and hospital operations. Access follows strict privacy regulations.

For UCLA Authorized Mednet users, institutional data access is granted. However, due to privacy constraints, UCLA EHR Data distribution is restricted outside these parameters.

# Installation and Setup

To replicate our results, install the MedBERT model from *Hugging Face*. We recommend using PyTorch 1.13. Ensure you use the specified seed for consistent train, validation, and test splits with our provided split function.

```python
from transformers import AutoTokenizer, AutoModel

# Load MedBERT from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
model = AutoModel.from_pretrained("Charangan/MedBERT")
...

# Split your dataset
train, validate, test = train_validate_test_split(data, seed=7)
```

More details on MedBERT can be found on its [Hugging Face page](https://huggingface.co/Charangan/MedBERT).

# Training Configuration and Execution

For training, ensure you have GPU access. Adjust the batch size based on your GPU memory; we used a batch size of 64. Note that changes in batch size might slightly affect the results. Set up your training environment by creating a `.json` configuration file and a `logs/` directory. Here's a sample structure for your configuration file:

```json
{
    "log_dir": "logs",
    "experiment_name": "Your_Experiment_Name",
    "batch": "Your_Batch_Size",
    "num_workers": 8,  // Enhances tokenization speed
    "data": "Path_to_Your_Data_File",
    "model": "./Charangan/MedBERT",  // Refer to the Installation section for model setup
    "mode": "multimodal",
    "gpu": "Your_GPU_ID",
    "task": "Your_Task",  // 'eddispo' or 'multitask'
    "epoch": "Your_Num_Epochs",  // Early stopping is implemented
    "dataset": "MIMIC"
}
```

To start the training, initiate a screen session and execute the following command:

```
python3 trainer.py -c train_config.json
```

Ensure you replace placeholder values with actual data specific to your setup.

# Inference Configuration and Execution

For running inference, you will need to set up a `.json` configuration file similar to the training setup but with additional fields specific to inference. Here's a sample structure:

```json
{
    "log_dir": "logs",
    "experiment_name": "Your_Experiment_Name",
    "batch": "Your_Batch_Size",
    "num_workers": 8,
    "data": "Path_to_Your_Data_File",
    "model": "./Charangan/MedBERT",
    "weights": "Path_to_Your_Model_Weights",
    "mode": "multimodal",
    "gpu": "Your_GPU_ID",
    "task": "Your_Task",
    "dataset": "MIMIC",
    "validation": "Validation_Mode"  // 'within' or 'across'
}
```

To initiate the inference process, start a screen session and run the following command:

```
python3 inference.py -c inference_config.json
```

# HuggingFace Model Weights

Model weights for MIMIC trained models can be found on the huggingface Website: [here](https://huggingface.co/Simonlee711/MEME)

# Citing
```
@article{lee2025clinical,
  title={Clinical decision support using pseudo-notes from multiple streams of EHR data},
  author={Lee, Simon A and Jain, Sujay and Chen, Alex and Ono, Kyoka and Biswas, Arabdha and Rudas, {\'A}kos and Fang, Jennifer and Chiang, Jeffrey N},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={394},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
