"""
Class of GatingNN or ComponentNN
"""
import torch
import torch.nn as nn
from ExpertWeights import ExpertWeights

class ComponentNN(nn.Module):
    def __init__(self, rng, input_x, num_experts, dim_layers, activation, weight_blend, keep_prob, batchSize, name,
                 FiLM=None):
        """

        :param rng: random seed for numpy
        :param input_x: input tensor of ComponentNN/GatingNN
        :param num_experts: number of experts
        :param dim_layers: dimension of each layer including the dimension of input and output
        :param activation: activation function of each layer
        :param weight_blend: blending weights from previous ComponentNN that used experts in
                             current Components NN.
                             Note that the VanillaNN can also be represented as Components NN with 1 Expert
        :param keep_prob: for drop out
        :param batchSize: for batch size
        :param name: for name of current component
        :param FiLM: Technique of FiLM, will not use this one in default
        """
        self.name = name

        """rng"""
        self.initialRNG = rng

        """input"""
        self.input = input_x

        """dropout"""
        self.keep_prob = keep_prob

        """"NN structure"""
        self.num_experts = num_experts
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers) - 1
        self.activation = activation
        self.batchSize = batchSize
        
        self.dp_list = []
        for _ in range(self.num_layers):
            self.dp_list.append(nn.Dropout(self.keep_prob))

    def initExperts(self):
        experts = []
        for i in range(self.num_layers):
            expert = ExpertWeights(self.initialRNG, (self.num_experts, self.dim_layers[i + 1], self.dim_layers[i]))
            experts.append(expert)
        return experts

    def buildNN(self, weight_blend, FiLM):
        return None

    def saveNN(self, sess, savepath, index_component):
        return None