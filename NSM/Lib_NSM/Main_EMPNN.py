import numpy as np
import torch
import sys

sys.path.append('../Lib_Expert')
sys.path.append('../Lib_Optimizer')
sys.path.append('../Lib_Opt')

from ComponentNN import ComponentNN
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
from NeuralNetwork import NeuralNetwork
from MLP import MLP

class MainNN(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self)
    