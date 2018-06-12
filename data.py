import torch
class Transition():
    def __init__(self,capacity):
        self.transitions = []
        self.capacity = capacity

    def store(self,old_state,action,reward,new_state,done):
        self.transitions.append((old_state,action,reward,new_state,done)) 
        if len(self.transitions) > self.capacity:
            self.transitions.pop(0) 

    def sample(self,sample_num,qnetwork):
        import numpy as np
        trans_num = len(self.transitions)
        sample_num = min(sample_num,trans_num)
        sample_idxs = np.random.choice(range(len(self.transitions)),sample_num,replace=False)


        sampled_transitions =  []
        for sample_idx in sample_idxs :
            sampled_transitions.append(self.transitions[sample_idx])

        

        ## to batch
        old_state_batch,new_state_batch,action_batch,reward_batch,dones = Batch(sampled_transitions)\
        .toTensor2d()
        
        # old_state_batch,new_state_batch,action_batch,reward_batch
        return old_state_batch,new_state_batch,action_batch,reward_batch,dones


class State():
    # k : observation number
    def __init__(self,k):
        self.k = k 
        assert self.k>=1
        self.history_list = [] # k state

    def init_state(self,first_observation):
        for _ in range(self.k-1):
            self.history_list.extend([-1.3,0.0,3])
        self.history_list.extend(first_observation)
        
    def update_state_by_observation(self,observation,action):
        if self.k > 1:
            self.history_list = self.history_list[3:]
            self.history_list.append(action)
        else :
            self.history_list = []
        self.history_list.extend(observation)

    def toTensor(self):
        # 1d tensor
        return torch.tensor(self.history_list).float()

    def display(self):
        pass
        #print('state')
        #print(self.history_list)

class Batch():
    def __init__(self,transitions):
        self.transitions = transitions


    def state2batch(self,x):
        return  list(map(lambda state:state.toTensor(),x))

    def toTensor2d(self):
        # state to batch 
        # action to batch
        # reward to batch

        old_states,actions,rewards,new_states,dones = \
        zip(*self.transitions)

        old_state_batch = torch.stack(self.state2batch(old_states),0)
        new_state_batch = torch.stack(self.state2batch(new_states),0)
        action_batch = torch.tensor(actions)
        reward_batch = torch.tensor(rewards)
        dones = torch.tensor(dones).gt(0).float()
  
      

        return old_state_batch,new_state_batch,action_batch,reward_batch,dones

        
            
    
