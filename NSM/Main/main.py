import sys

sys.path.append('../Lib_NSM')
import numpy as np
import torch

from Main_EMPNN import MainNN

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

# Path Setting
load_path = '../../data'
save_path = '../../trained'
type_normalization = 0
name_model = "NSM"

def main():
    model = MainNN()

if __name__ == '__main__':
    main()    