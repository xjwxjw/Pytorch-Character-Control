# Vanilla MLP
import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, rng, dim_layers, activation, keep_prob):
        super(MLP, self).__init__()
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

        # for i in range(self.num_layer):
        self.fc0 = nn.Linear(in_features = self.dim_layers[0],\
                                out_features = self.dim_layers[1],\
                                bias = True)
        self.fc1 = nn.Linear(in_features = self.dim_layers[1],\
                                out_features = self.dim_layers[2],\
                                bias = True)
        
        # for i in range(self.num_layer):
        self.dp0 = nn.Dropout(1 - self.keep_prob)
        self.dp1 = nn.Dropout(1 - self.keep_prob)
        
        # for i in range(self.num_layer):
        self.ac0 = nn.ELU()
        self.ac1 = nn.ELU()
        
    def forward(self, input_):
        x = input_
        x = self.ac0(self.fc0(self.dp0(x)))
        x = self.ac1(self.fc1(self.dp1(x)))
        return x
