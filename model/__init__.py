from model.networks import TransformerNetwork_3x
import torch.nn as nn
import torch.nn.functional as F


def get_autoencoder(config):
    # return AutoEncoder3x(config.mot_en_channels, config.body_en_channels,
    #                      config.view_en_channels, config.de_channels)
    return TransformerNetwork_3x()
