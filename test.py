import gym, random,copy,os,torch
from model import   QNetwork
from data import State,Transition
from torch import optim
import pickle as pkl

max_episode_steps = 200
#gym.envs.register(
#    id='MountainCarExtraLong-v0',
#    entry_point='gym.envs.classic_control:MountainCarEnv',
#    max_episode_steps=max_episode_steps
#)
env = gym.make('MountainCar-v0')
#env = gym.make('MountainCar-v0')
# load parameter
load_flag = False
save_dir = './model/h2lr2'
model_dir = os.path.join(save_dir,'model_300.pkl')
#RL LOOP
lr = 0.0001
episode_num = 1000
history_num =2
epsilon = 0.9
gamma = 0.99
history_path = 'history_report/%.3f_%d.pkl'%(lr,history_num)
# parameter of qnetwork
input_dim,h1_dim,h2_dim,h3_dim,action_num = (history_num-1)*3+2,128,64,32,3
net_hypers = [input_dim,h1_dim,h2_dim,h3_dim,action_num]    
batch_size = 256
capacity = 10000
transition = Transition(capacity)
save_model_per_episode = 10
save_history_per_episode = 10
if load_flag:
    checkpoint = torch.load(model_dir)
    qnetwork = QNetwork(*checkpoint['model_hyper'])
    qnetwork.load_state_dict(checkpoint['model']) 
    epsilon = 0.05
else:
    qnetwork =  QNetwork(*net_hypers)
    
optimizer = optim.Adam(qnetwork.parameters(),lr=lr) 
if load_flag:
    optimizer.load_state_dict(checkpoint['optimzer'])




loss_history = []
reward_sum_history = []
for episode in range(episode_num):
    if (episode+1) % 2 == 0:
        if epsilon >0.05:
            epsilon = epsilon*0.99
 
    print('episode %d'%(episode))
    observation = env.reset()
    #initialize state
    state = State(history_num)
    state.init_state(observation)
    done = False
    final_transition = None
    loss_sum = 0
    reward_sum = 0

    for t in range(max_episode_steps):
        if done:
            break
        env.render()
        state.display()
        # select a action
        p = random.random()
        if  p < epsilon:
            action = env.action_space.sample()
        else:
            action = qnetwork.decide_action(state.toTensor().view(1,-1))
            # to scalar and then to int
            assert list(action.shape) == [1]
            action = action.sum().item()
        
        observation, reward, done, info = env.step(action)

        
        if done:  
            # -1 reward for all step
            avg_loss = loss_sum/max_episode_steps
            print('done')
            print(avg_loss)
            loss_history.append(avg_loss)
            reward_sum_history.append(reward_sum)



        old_state = copy.deepcopy(state) 
        state.update_state_by_observation(observation,action)
        new_state =  state

       
        tr = old_state,action,reward,new_state,done

        transition.store(*tr)
        
        S1,S2,A,R,D = transition.sample(batch_size,qnetwork)
     
        Y = R+(1-D)*gamma*qnetwork.qvalue(S2,A)
        diff = ( Y  - qnetwork.qvalue(S1,A))
        loss = (diff*diff).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum = loss_sum+loss.item()
        reward_sum = reward_sum+reward

        if  (episode+1) % save_model_per_episode == 0 :
            torch.save({'episode':str( episode),'model':qnetwork.state_dict(),\
                'optimzer':optimizer.state_dict(),'model_hyper':net_hypers},
                 os.path.join(save_dir,'%.3f_%d_model_%d.pkl'%(lr,history_num,episode)))

        if (episode+1) % save_history_per_episode == 0: 
            with open(history_path,'wb') as f:
                pkl.dump(( loss_history,reward_sum_history),f)   