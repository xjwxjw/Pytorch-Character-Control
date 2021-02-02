# Basic Class for MainNN
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

import torch
sys.path.append('../Lib_Utils')
import Utils as utils

class LocoMotionData(Dataset):
    def __init__(self, load_path, save_path, type_normalize=0):
        # for reproducibility
        self.rng = np.random.RandomState(1234)
        torch.manual_seed(1234) 
        self.load_path = load_path
        self.save_path = save_path
        self.type_normalize = type_normalize
        self.ProcessData(self.load_path, self.save_path, self.type_normalize)

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        return torch.from_numpy(self.input_data[index]), torch.from_numpy(self.output_data[index])

    def ProcessData(self, load_path, save_path, type_normalize=0):
        """
        :param load_path: path of data
        :param save_path: path for saving
        :param type_normalize: how to normalize the data
                0. (default normalization) load mean and std from txt and (data-mean)/std
                1. coming soon

        :return: a. build savepath
                 b. laod data
                 c. get the input dim, output dim and data size
        """
        self.save_path = save_path
        utils.build_path([save_path])
        if (type_normalize == 0):
            self.input_data, self.input_mean, self.input_std = utils.Normalize(
                np.float32(np.loadtxt(load_path + '/Input.txt')),
                np.float32(np.loadtxt(load_path + '/InputNorm.txt')),
                savefile=save_path + '/X')
            self.output_data, self.output_mean, self.output_std = utils.Normalize(
                np.float32(np.loadtxt(load_path + '/Output.txt')),
                np.float32(np.loadtxt(load_path + '/OutputNorm.txt')),
                savefile=save_path + '/Y')
        else:
            print("Coming Soon!!!")
        self.input_dim = self.input_data.shape[1]
        self.output_dim = self.output_data.shape[1]
        self.data_size = self.input_data.shape[0]
        print("Data is Processed")