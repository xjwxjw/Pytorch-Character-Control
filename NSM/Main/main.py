import sys

sys.path.append('../Lib_NSM')
import numpy as np
import torch

from NSMNN import NSMNN
from LocoMotionData import LocoMotionData
from torch.utils.data import DataLoader

# Tuning Settings
hidden_size_Gating = 512
hidden_size_Main = 512
keep_prob_components = 0.7

# Index
start_pose = 0
start_goal = 419
start_environment = 575
start_interaction = 2609
start_gating = 4657
dim_gating = 650

# Training Settings:
num_epoch = 150
batch_size = 32

# Path Setting
load_path = '../../data'
save_path = '../../trained'
type_normalization = 0
name_model = "NSM"


def main():
    ## construct dataloader
    LocoData = LocoMotionData(load_path, save_path, type_normalization)
    LocoDataLoader = DataLoader(LocoData, batch_size = batch_size, shuffle = True, num_workers = 4)

    ## build model
    ## input format(pose:0-347, trajectory: 349-399, goal:400-477)
    model = NSMNN()

    ## start training
    for epoch in range(num_epoch):
        for idx, (input_, output_) in enumerate(LocoDataLoader):
            # print(epoch, idx, input_.size(), output_.size()) 
            pass
                 
    
    

if __name__ == '__main__':
    main()    
