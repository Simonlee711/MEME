

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

class Embeddings(nn.Module):
    """
    Construct the embeddings for admission, meds, labs, codes and add it to position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        token_lim = config.cc_len  # token limits - 512
        num_lab = config.lab_len  # number of labs

        # Linear Layer Embedding
        self.admissions_embeddings = Linear(768, config.hidden_size)  
        self.labs_embeddings = Linear(768, config.hidden_size)  
        self.meds_embeddings = Linear(768, config.hidden_size)  
        self.codes_embeddings = Linear(768, config.hidden_size)  
        self.time_embeddings = Linear(768,config.hidden_size)
        
        # Positional Embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.pe_labs = nn.Parameter(torch.zeros(1, num_lab, config.hidden_size))
        self.pe_meds = nn.Parameter(torch.zeros(1, token_lim, config.hidden_size))
        self.pe_admissions = nn.Parameter(torch.zeros(1, token_lim, config.hidden_size))
        self.pe_codes = nn.Parameter(torch.zeros(1, token_lim, config.hidden_size))

        # Need a cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # add some dropout layers to mitigate overfitting
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_admissions = Dropout(config.transformer["dropout_rate"])
        self.dropout_labs = Dropout(config.transformer["dropout_rate"])
        self.dropout_meds = Dropout(config.transformer["dropout_rate"])
        self.dropout_codes = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, admissions, labs, meds, codes):
        """
        Feed Forward network for the embedding layer
        """
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        admissions = self.admissions_embeddings(admissions)
        labs = self.labs_embeddings(labs)
        meds = self.meds_embeddings(meds)
        codes = self.codes_embeddings(codes)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        admissions_embeddings = admissions + self.pe_admissions
        labs_embeddings = labs + self.pe_labs
        meds_embeddings = meds + self.pe_meds
        codes_embeddings = codes + self.pe_codes

        embeddings = self.dropout(embeddings)
        admissions_embeddings = self.dropout_cc(admissions_embeddings)
        labs_embeddings = self.dropout_lab(labs_embeddings)
        meds_embeddings = self.dropout_meds(meds_embeddings)
        codes_embeddings = self.dropout_codes(codes_embeddings)

        # concatenate all the input
        final = embeddings + admissions_embeddings + labs_embeddings + meds_embeddings + codes_embeddings
        return 