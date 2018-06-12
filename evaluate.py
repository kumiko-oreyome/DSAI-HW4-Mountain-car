from data import State
import gym, random,copy,os,torch
import pickle as pkl
import torch
from model import   QNetwork 

def evaluate(model_path,history_num,max_episode_steps,episode_num,result_save_path):
    checkpoint = torch.load(model_path)
    qnetwork = QNetwork(*checkpoint['model_hyper'])
    qnetwork.load_state_dict(checkpoint['model']) 
    

    env = gym.make('MountainCar-v0')
    test_success_history = []
    test_reward_history = []
    for episode in range(episode_num):
        print('episode %d'%(episode))
        observation = env.reset()
        #initialize state
        state = State(history_num)
        state.init_state(observation)
        done = False
        reward_sum = 0
    
        for t in range(max_episode_steps):
            env.render()
            state.display()
            # select a action with max q value action
            action = qnetwork.decide_action(state.toTensor().view(1,-1))
            action = action.sum().item() 
            observation, reward, done, info = env.step(action)
            reward_sum = reward_sum+reward

            if done: 
                print('done')
                print(reward_sum)
                success = False
                if observation[0]>=0.5:
                    success = True
                test_success_history.append(success)
                test_reward_history.append(reward_sum)
                break

            state.update_state_by_observation(observation,action)
            


    print('- '*100)
    print('save to %s'%(result_save_path))  
    with open(result_save_path,'wb') as f:
        pkl.dump((test_success_history,test_reward_history),f)


history_num = 2
lr = 0.001
final_episode = 999
model_path ='./model/%.3f_%d_model_%d.pkl'%(lr,history_num,final_episode)

evaluate(model_path,history_num,200,100,'evaluate/%.3f_%d_result_%d.pkl'%(lr,history_num,final_episode))