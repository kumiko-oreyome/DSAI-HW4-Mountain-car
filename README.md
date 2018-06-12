# DSAI HW4 : mountain car v0

#### Model
In this homework,I implements a deep Q networks , which is a 4 layer DNN and outputs 2 action(left/right) Q-values.


#### state
I design the state which consist of latest n observation history , n is hyperparameter

o: observation (p,v)  
&emsp; p is position,v is velocity
a : action

if n = 1 :
&emsp;&emsp;the state is (p,v)
if n > 1:
&emsp; t : current time step of an epsidoe
&emsp;&emsp;the state is previous n-1 observation,action plus current observation
  (p[t-n-1],o[t-n-1],a[t-n-1].....p[t-2],v[t-2],a[t-2],p[t-1],v[t-1],a[t-1],p[t],v[t])

#### reward
-1 , same as the gym environment 

In the following report , I will show DQN's performance under differnet learning rate and state num n.

lr is learning rate
n is observation history num

I try lr=0.001,0.0001 and n = 1,2
4 agents is trained and evaluated.





#### Analyze
I train lots of network but all fail to reach the goal on the right mountain in 200 steps (training with 1000 episodes , max step 200 each epsiode).


It is a hard problem,if only using 200 max steps,It seldom success reach the position > 0.5 , so always get reward -1 , which makes Q-network hard to learn how to success,I it thinks all the action as worse as other action because they all get -1 rewards on whole trajectory, so it took long time to train a success agent.

I train a n=1 , lr=0.001 agent with 10000 epsidoes,it success in the training procedure between 2000~2500,but after 2500 it fails to solve the mountain car problem .It seems like DQN is not a stable kearning algorithm.


### Reports

all the result figure can be found in **analyze.ipynb** notebook.

##### evaluation results
######  &emsp; loss:root mean square error between  r+gamma*Q(s',a') and Q(s,a)
&emsp;&emsp; loss of 4 trainig procedure get lower and lower 
######  &emsp; reward sum
&emsp;&emsp;most of the reward sum is -199 (the last step I forget to add to reward sum)




##### evaluation results
Here is the evalation result of each agent .

Each test run 100 episode.
If agent reach position > 0.5 it success.
Each episode I count the sum of reward in that episode.

Following show the  success rate and average reward sum of 100 episodes.
Because they all fail to solve the mountain car problem.Success rate is 0 and average reward sum is -200.


| History number | Learning rate | Success rate | Avg reward sum
| ------ | ------ | ------ | ------ |
| 1 | 0.001 | 0.0 | -200
| 1 | 0.0001 | 0.0 | -200
| 2| 0.001 | 0.0 | -200
| 2 | 0.0001 | 0.0 | -200




#### Answer of 3 questions
1. What kind of RL algorithms did you use? value-based, policy-based, model-based? why?  
DQN,value-based,because Q learning try to find the opitmal value(discounted total reward) under a state.

2. This algorithms is off-policy or on-policy? why? (10%)
  off-policy,because DQN  using replay memory to update the policy ,which is not the current action the agent take.

3. How does your algorithm solve the correlation problem in the same MDP? (10%) I use  **replay memory** to solve the correlation problem,which will remember previous history and t training batch comes from it so learning isn't favor of  latest states.
















