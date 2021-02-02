# Vanilla MLP
import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, rng, input_x, dim_layers, activation, keep_prob, name):
        """
        feed forward network that is usually used as encoder/decoder
        :param rng: random seed for initialization
        :param input_x: input tensor (batch_size * dim)
        :param dim_layers: list of int values for dim
        :param activation: list of activation functions
        :param keep_prob: keep prob
        :param name: name
        """
        pass