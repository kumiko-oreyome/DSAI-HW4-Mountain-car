import torch
import torch.nn as nn
import torch.nn.functional as F
        


class QNetwork(nn.Module):
    def __init__(self,input_dim,h1_dim,h2_dim,h3_dim,action_num):
        super(QNetwork, self).__init__()
        self.L1 = self.__create_linear_layer(input_dim,h1_dim)
        self.L2 = self.__create_linear_layer(h1_dim,h2_dim)
        self.L3 = self.__create_linear_layer(h2_dim,h3_dim)
        self.value_layer = self.__create_linear_layer(h3_dim,action_num)
        
    def __create_linear_layer(self,d1,d2):
        l  = nn.Linear(d1,d2)
        nn.init.xavier_normal_(l.weight)
        return l
    
    # state : N * 14 (3*4+2)
    # output : N *3
    def forward(self,state):
        z1 = self.L1(state)
        h1 = F.relu(z1)
        z2 = self.L2(h1)
        h2 = F.relu(z2)
        z3 = self.L3(h2)
        h3 = F.relu(z3)
        # N*3
        z4 = self.value_layer(h3)
        return z4


    # calculate q value on a batch
    # state : N * 14 (3*4+2)
    # output : N 
    def qvalue(self,state,actions):
        score = self.forward(state)
        qvalues = torch.gather(score,1,actions.view(-1,1))
        qvalues = qvalues.squeeze(1)
        return qvalues


    # output shape N
    def qstar(self,state):
        score = self.forward(state)
        qstar_value,_ = torch.max(score,1)
        return qstar_value

    # return shape N 
    def decide_action(self,state):
        score = self.forward(state)
        actions = torch.argmax(score,1)
        return actions