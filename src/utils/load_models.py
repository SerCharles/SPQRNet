'''
Description: load models
Author:Charles Shen
Date:8/29/2020
'''

import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import constants
from models.encoder import Encoder
from models.decoder import Decoder
from models.PCN import PCN

def init_trained_model(base_dir, model_dir, device):
    '''
        description: load trained model
        variable: base_dir, model_dir, device
        return: encoder_anchor, encoder_positive, decoder
    '''
    encoder_anchor = Encoder()
    encoder_positive = Encoder()
    decoder = Decoder()
    encoder_anchor.load_state_dict(torch.load(os.path.join(base_dir, model_dir, constants.model_name_partial),  map_location='cpu'))
    encoder_positive.load_state_dict(torch.load(os.path.join(base_dir, model_dir, constants.model_name_complete),  map_location='cpu'))
    decoder.load_state_dict(torch.load(os.path.join(base_dir, model_dir, constants.model_name_decoder),  map_location='cpu'))
    if device:
        encoder_anchor.to(device)
        encoder_positive.to(device)
        decoder.to(device)
    return encoder_anchor, encoder_positive, decoder

def init_trained_PCN(base_dir, model_name, device):
    '''
        description: load trained PCN
        variable: base dir, model name, device
        return: model
    '''
    model = PCN()
    model.load_state_dict(torch.load(os.path.join(base_dir, model_name, constants.model_name_PCN),  map_location='cpu'))
    if device:
        model.to(device)
    return model