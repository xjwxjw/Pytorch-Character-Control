import numpy as np
import torch
import torch.nn as nn

class ExpertWeights(nn.Module):
    def __init__(self, rng, shape, name = None):
        """rng"""
        self.initialRNG = rng

        """shape"""
        self.weight_shape = shape  # 4/8 * out * in
        self.bias_shape = (shape[0], shape[1], 1)  # 4/8 * out * 1

        """alpha and beta"""
        self.alpha = torch.Tensor(self.initial_alpha(), dtype=torch.float64).cuda().requires_grad_()
        self.beta = torch.Tensor(self.initial_beta(), dtype=torch.float64).cuda().requires_grad_()

    """initialize parameters for experts i.e. alpha and beta"""

    def initial_alpha_np(self):
        shape = self.weight_shape
        rng = self.initialRNG
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return alpha
    
    def initial_alpha(self):
        return self.initial_alpha_np()

    def initial_beta_np(self):
        return self.initial_beta()
    
    def get_NNweight(self, controlweights, batch_size):
        a = self.alpha.unsqueeze(1) # 4*out*in   -> 4*1*out*in
        a = a.repeat(1, batch_size, 1, 1)  # 4*1*out*in -> 4*?*out*in
        w = controlweights.unsqueeze(-1).unsqueeze(-1)  # 4*?        -> 4*?*1*1
        r = w * a  # 4*?*1*1 m 4*?*out*in
        return torch.sum(r, axis=0)  # ?*out*in

    def get_NNbias(self, controlweights, batch_size):
        b = self.beta.unsqueeze(1)  # 4*out*1   -> 4*1*out*1
        b = b.repeat(1, batch_size, 1, 1)  # 4*1*out*1 -> 4*?*out*1
        w = controlweights.unsqueeze(-1).unsqueeze(-1)  # 4*?        -> 4*?*1*1
        r = w * b  # 4*?*1*1 m 4*?*out*1
        return torch.sum(r, axis=0)  # ?*out*1
