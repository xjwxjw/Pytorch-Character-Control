import sys

sys.path.append('../Lib_NSM')
import numpy as np
import torch

from NSMNN import NSMNN
from LocoMotionData import LocoMotionData
from torch.utils.data import DataLoader
import torch.optim as optim
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
data_load_path = '../../data'
model_load_path = '../../trained'
type_normalization = 0
name_model = "NSM"


def main():
    ## construct dataloader
    LocoData = LocoMotionData(data_load_path, model_load_path, type_normalization)
    LocoDataLoader = DataLoader(LocoData, batch_size = batch_size, shuffle = True, num_workers = 4)

    ## build model
    ## input format(pose:0-347, trajectory: 349-399, goal:400-477, gate: 478-555)
    model = NSMNN().cuda()
    ## construct optimizer
    # for item in model.state_dict():
    #     print(item)
    # optimizer = optim.AdamW(params = model.parameters(), lr = 0.0001, weight_decay=0.00025)
    model.eval()
    model.load_state_dict(torch.load(model_load_path + '/model.pth'))
    # torch.save(model.state_dict(), '../../trained/model.pth')
    ## start training
    for epoch in range(num_epoch):
        # model.train()
        # loss_list = []
        for idx, (input_, output_) in enumerate(LocoDataLoader):
            # print(epoch, idx, input_.size(), output_.size()) 
            input_ = input_.cuda()
            output_ = output_.cuda()
            pred_ = model(input_)
            # loss = torch.mean(torch.abs(output_ - pred_))

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # loss_list.append(loss.item())
    #     print(epoch, np.mean(loss_list))
    # torch.save(model.state_dict(), '../../trained/model.pth')

if __name__ == '__main__':
    main()    
