# Vanilla MLP
import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, rng, dim_layers, activation, keep_prob):
        super().__init__()
        """
        feed forward network that is usually used as encoder/decoder
        :param rng: random seed for initialization
        :param input_x: input tensor (batch_size * dim)
        :param dim_layers: list of int values for dim
        :param activation: list of activation functions
        :param keep_prob: keep prob
        :param name: name
        """
        
        """rng"""
        self.initialRNG = rng

        """dropout"""
        self.keep_prob = keep_prob

        """"NN structure"""
        self.dim_layers = dim_layers
        self.activation = activation
        self.num_layer = len(activation)

        assert self.num_layer + 1 == len(self.dim_layers)

        self.fc_list = []
        for i in range(self.num_layer):
            fc_layer = nn.Linear(in_features = self.dim_layers[i],\
                                 out_features = self.dim_layers[i+1],\
                                 bias = True)
            self.fc_list.append(fc_layer)
        
        self.dp_list = []
        for i in range(self.num_layer):
            dp_layer = nn.Dropout(1 - self.keep_prob)
            self.dp_list.append(dp_layer)
        
        self.ac_list = []
        for i in range(self.num_layer):
            ac_layer = nn.ELU()
            self.ac_list.append(ac_layer)
        
    def forward(self, input_):
        x = input_
        for i in range(self.num_layer):
            x = self.dp_list[i](x)
            x = self.fc_list[i](x)
            x = self.ac_list[i](x)
        return x
