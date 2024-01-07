import numpy as np

# from sagemaker import get_execution_role
# import boto3
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
import pandas as pd
import json
import pickle
from transformers import AdamW, get_scheduler
from datasets import load_metric

import numpy as np

# from sagemaker import get_execution_role
# import boto3
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
import pandas as pd
import json
import pickle
from transformers import AdamW, get_scheduler
from datasets import load_metric


class ED_encoder(nn.Module):
    """
    A task-specific custom transformer model for predicting ED Disposition. 
    This model loads a pre-trained transformer model and adds a new dropout 
    and linear layer at the end for fine-tuning and prediction on specific tasks.
    """

    def __init__(self, checkpoint, num_labels, freeze=True):
        """
        Args:
            checkpoint (str): The name of the pre-trained model or path to the model weights.
            num_labels (int): The number of output labels in the final classification layer.
        """
        super(ED_encoder, self).__init__()
        self.num_labels = num_labels  # number of labels for classifier

        # checkpoint is the model name
        self.model = model = AutoModel.from_pretrained(
            checkpoint,
            config=AutoConfig.from_pretrained(
                checkpoint, output_attention=True, output_hidden_state=True
            ),
        )
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor, optional): Tensor of input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): Tensor for attention masks. Defaults to None.
            labels (torch.Tensor, optional): Tensor for labels. Defaults to None.
            
        Returns:
            TokenClassifierOutput: A named tuple with the following fields:
            - loss (torch.FloatTensor of shape (1,), optional, returned when label_ids is provided) – Classification loss.
            - logits (torch.FloatTensor of shape (batch_size, num_labels)) – Classification scores before SoftMax.
            - hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
            - attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) – Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        # calls on the Automodel to deploy correct model - in our case distilled-bert-uncased
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # retrieves the last hidden state
        last_hidden_state = outputs[0]

        return last_hidden_state  # The embedding


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels=3):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.num_labels = 2  # number of labels for classifier
        self.dropout = nn.Dropout(0.3)  # to prevent overfitting
        self.classifier = nn.Linear(
            768, num_labels
        )  # FC Layer - takes in a 768 token vector and is a Linear classifier with n labels
        self.dense_layer = nn.Linear(768, 768)
        self.relu = nn.ReLU()  # ReLU non-linearity
        self.loss_func = (
            nn.BCEWithLogitsLoss()
        )  # Change this if it becomes more than binary classification

    def binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        loss = 0
        for i, label in enumerate(["further_discharge", "mortality", "ICU"]):
            loss -= y_true[:, i] * torch.log(y_pred[:, i]) + (
                1 - y_true[:, i]
            ) * torch.log(1 - y_pred[:, i])
        return loss.mean()

    def forward(self, x, labels=None):
        # typical self attention workflow
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)

        # include dropout from constructor to feed forward network
        new_vec = self.dense_layer(weighted)
        new_vec = self.relu(new_vec)  # Apply ReLU activation

        sequence_outputs = self.dropout(new_vec)
        # finally add linear layer from input
        logits = self.classifier(sequence_outputs[:, 0, :].view(-1, 768))

        loss = None
        if labels is not None:
            loss = self.binary_cross_entropy(labels, torch.sigmoid(logits))

            # TokenClassifierOutput - returns predicted label
            return TokenClassifierOutput(loss=loss, logits=logits)

        else:
            return logits


class ED_classifier(nn.Module):
    def __init__(
        self, checkpoint, num_labels=2, input_dim=768, modalities=None, freeze=True
    ):
        super(ED_classifier, self).__init__()
        self.encoder = ED_encoder(
            checkpoint=checkpoint, num_labels=num_labels, freeze=freeze
        )
        self.predictor = Classifier(input_dim=input_dim, num_labels=num_labels)
        assert modalities is not None, "Number of modalities missing"
        self.modalities = modalities

    def forward(self, input_ids, attention_mask, label=None):
        # input_ids: dictionary of the batch
        # attention_mask: dictionary of the batch
        embedding = []
        min_batch_size = float("inf")
        for modality in range(self.modalities):
            embed = self.encoder(input_ids[modality], attention_mask[modality], label)
            embedding.append(embed)
            min_batch_size = min(
                min_batch_size, embed.size(0)
            )  # Find the smallest batch size

        # Ensuring all embeddings have the same batch size
        uniform_embeddings = [e[:min_batch_size] for e in embedding]

        # Concatenating embeddings
        unified_embedding = torch.cat(
            uniform_embeddings, 1
        )  # concatenates embeddings on the second dimension
        outputs = self.predictor(unified_embedding, label)
        return outputs


class SingleModPredictor(nn.Module):
    """
    A task-specific custom transformer model for predicting ED Disposition. 
    This model loads a pre-trained transformer model and adds a new dropout 
    and linear layer at the end for fine-tuning and prediction on specific tasks.
    """

    def __init__(self, checkpoint, num_labels, freeze=True):
        """
        Args:
            checkpoint (str): The name of the pre-trained model or path to the model weights.
            num_labels (int): The number of output labels in the final classification layer.
        """
        super(SingleModPredictor, self).__init__()
        self.num_labels = num_labels  # number of labels for classifier

        # checkpoint is the model name
        self.model = model = AutoModel.from_pretrained(
            checkpoint,
            config=AutoConfig.from_pretrained(
                checkpoint, output_attention=True, output_hidden_state=True
            ),
        )
        # New Layer
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)  # FC Layer
        self.loss_func = nn.BCEWithLogitsLoss()
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        loss = 0
        for i, label in enumerate(["further_discharge", "mortality", "ICU"]):
            loss -= y_true[:, i] * torch.log(y_pred[:, i]) + (
                1 - y_true[:, i]
            ) * torch.log(1 - y_pred[:, i])
        return loss.mean()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor, optional): Tensor of input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): Tensor for attention masks. Defaults to None.
            labels (torch.Tensor, optional): Tensor for labels. Defaults to None.
            
        Returns:
            TokenClassifierOutput: A named tuple with the following fields:
            - loss (torch.FloatTensor of shape (1,), optional, returned when label_ids is provided) – Classification loss.
            - logits (torch.FloatTensor of shape (batch_size, num_labels)) – Classification scores before SoftMax.
            - hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
            - attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) – Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]

        sequence_outputs = self.dropout(last_hidden_state)

        logits = self.classifier(sequence_outputs[:, 0, :].view(-1, 768))

        loss = None
        if labels is not None:
            loss = self.binary_cross_entropy(labels, torch.sigmoid(logits))

            # TokenClassifierOutput
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
