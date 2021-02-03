"""
Class of GatingNN or ComponentNN
"""
import torch
import torch.nn as nn
from ExpertWeights import ExpertWeights

class ComponentNN(nn.Module):
    def __init__(self, rng, num_experts, dim_layers, keep_prob, batchSize,
                 FiLM=None):
        super(ComponentNN, self).__init__()
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

        """rng"""
        self.initialRNG = rng

        """dropout"""
        self.keep_prob = keep_prob

        """"NN structure"""
        self.num_experts = num_experts
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers) - 1
        self.batchSize = batchSize
        
        self.dp_list = []
        for _ in range(self.num_layers):
            self.dp_list.append(nn.Dropout(self.keep_prob))

        ## Construct the network
        self.experts = self.initExperts()
        
        self.dp_list = []
        self.ac_list = []
        for i in range(self.num_layers):
            dp_layer = nn.Dropout(self.keep_prob)
            self.dp_list.append(dp_layer)
            if i < (self.num_layers - 1):
                ac_layer = nn.ELU()
            else:
                ## there should not be always softmax !!!
                ac_layer = nn.Softmax(-1)
            self.ac_list.append(ac_layer)
        
    def forward(self, input_, batch_size, weight_blend = None, final_ac = False):
        H = input_.unsqueeze(-1)
        H = self.dp_list[0](H)
        for i in range(self.num_layers):
            if weight_blend is not None:
                w = self.experts[i].get_NNweight(weight_blend, batch_size)
                b = self.experts[i].get_NNbias(weight_blend, batch_size)
            else:
                w = self.experts[i].get_NNweight(torch.ones((1, batch_size)).cuda(), batch_size)
                b = self.experts[i].get_NNbias(torch.ones((1, batch_size)).cuda(), batch_size)
            H = torch.matmul(w, H) + b
            if i < (self.num_layers - 1):
                H = self.ac_list[i](H)
                H = self.dp_list[i](H)
            else:
                H = H.squeeze(-1)
                if final_ac:
                    H = self.ac_list[i](H)
        return H
        

    def initExperts(self):
        experts = []
        self.expert0 = ExpertWeights(self.initialRNG, (self.num_experts, self.dim_layers[1], self.dim_layers[0]))
        experts.append(self.expert0)
        self.expert1 = ExpertWeights(self.initialRNG, (self.num_experts, self.dim_layers[2], self.dim_layers[1]))
        experts.append(self.expert1)
        self.expert2 = ExpertWeights(self.initialRNG, (self.num_experts, self.dim_layers[3], self.dim_layers[2]))
        experts.append(self.expert2)
        return experts

    def buildNN(self, weight_blend, FiLM):
        return None

    def saveNN(self, sess, savepath, index_component):
        return None

if __name__ == '__main__':
    expert_components = [1, 10]
    dim_layers = [78, 512, 512, 10]
    model = ComponentNN(1234, 1, dim_layers, 0.7, 32)
    x = torch.zeros((32, 78))
    x = model(x)
    print(x.size())
