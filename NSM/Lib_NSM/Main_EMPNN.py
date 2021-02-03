import numpy as np
import torch
import sys

sys.path.append('../Lib_Expert')
sys.path.append('../Lib_Optimizer')
sys.path.append('../Lib_Opt')

from ComponentNN import ComponentNN
from MLP import MLP
import torch.nn as nn

class NSMNN(nn.Module):
    def __init__(self, hidden_size_Gating = 512,\
                       hidden_size_Main = 512,\
                       keep_prob_components = 0.7,\
                       start_pose = 0,\
                       dim_pose = 400,\
                       start_goal = 400,\
                       dim_goal = 78,\
                       start_environment = -1,\
                       dim_environment = 0,\
                       start_interaction = -1,\
                       dim_interaction = 0,\
                       start_gating = 478,\
                       dim_gating = 78):
        super(NSMNN, self).__init__()
        self.hidden_size_Gating = hidden_size_Gating
        self.hidden_size_Main = hidden_size_Main
        self.keep_prob_components = keep_prob_components
        self.start_pose = start_pose
        self.start_goal = start_goal
        self.start_environment = start_environment
        self.start_interaction = start_interaction
        self.start_gating = start_gating
        self.dim_gating = dim_gating

        ## For network structure
        self.expert_components = [1, 10]
        self.act_components = [
                                [nn.ELU(), nn.ELU(), nn.Softmax()],
                                [nn.ELU(), nn.ELU(), 0]
                            ]
        self.dim_components = [
                                [hidden_size_Gating, hidden_size_Gating],
                                [hidden_size_Main, hidden_size_Main]
                            ]
        self.input_components = [
                                np.arange(self.start_gating, self.start_gating + self.dim_gating),
                                []
                            ]
    
        ## For encoder
        self.index_encoders = [
                                np.arange(start_pose, start_pose + dim_pose),
                                np.arange(start_goal, start_goal + dim_goal)
                            ]
        self.dim_encoders = [
                                [512, 512],
                                [128, 128]
                            ]

        self.activation_encoders = [
                                [nn.ELU(), nn.ELU()],
                                [nn.ELU(), nn.ELU()]
                            ]

        if self.start_environment > -1:
            self.index_encoders.append(np.arange(start_environment, start_environment + dim_environment))
            self.dim_encoders.append([512, 512])
            self.activation_encoders.append([nn.ELU(), nn.ELU()])
        
        if self.start_interaction > -1:
            self.index_encoders.append(np.arange(start_interaction, start_interaction + dim_interaction))
            self.dim_encoders.append([512, 512])
            self.activation_encoders.append([nn.ELU(), nn.ELU()])
                                
        assert len(self.index_encoders) == \
               len(self.dim_encoders) == \
               len(self.activation_encoders)

        self.encoder_list = []
        encoder_dims = [len(self.index_encoders[0])] + self.dim_encoders[0]
        self.encoder0 = MLP(1234, encoder_dims, self.activation_encoders[0], 0.7)
        encoder_dims = [len(self.index_encoders[1])] + self.dim_encoders[1]
        self.encoder1 = MLP(1234, encoder_dims, self.activation_encoders[1], 0.7)
        
        ## For Gating Network
        dim_layers = [78, 512, 512, 10]
        self.gating_NN = ComponentNN(1234, 1, dim_layers, 0.7, 32)

        ## For motion prediction network
        dim_layers = [640, 512, 512, 578]
        self.motion_prediction_NN = ComponentNN(1234, 10, dim_layers, 0.7, 32)

    def forward(self, input_):
        self.num_encoders = len(self.index_encoders)
        enc_feat_list = []
        enc_feat = self.encoder0(input_[:, self.index_encoders[0]])
        enc_feat_list.append(enc_feat)
        enc_feat = self.encoder1(input_[:, self.index_encoders[1]])
        enc_feat_list.append(enc_feat)
        weight_blend = self.gating_NN(input_[:, self.start_gating:self.start_gating+self.dim_gating], input_.size(0), final_ac = True)
        weight_blend = weight_blend.permute(1, 0)
        enc_feat = torch.cat(enc_feat_list, -1)
        pred = self.motion_prediction_NN(enc_feat, input_.size(0), weight_blend, final_ac = True)
        return pred

if __name__ == '__main__':
    model = NSMNN()
    x = torch.zeros((32, 556))
    x = model(x)
