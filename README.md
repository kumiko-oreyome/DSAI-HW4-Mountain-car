# DSAI HW4 : mountain car v0

#### Agent
In this homework,I implemented a deep Q networks , which is a 4 layer DNN with relu acivation function and outputs Q-value of 2 actions(left/right).


#### state
I designed the state which consist of latest n observation history and actions, n is hyperparameter.


Here is the details:

o: observation (p,v)  
&emsp; p is position,v is velocity
a : action
n: history number mentioned above.

if n = 1 :
&emsp;&emsp;the state is (p,v) 
if n > 1:
&emsp; t : current time step of an epsidoe
&emsp;&emsp;the state is previous n-1 observation,action plus current observation
  (p[t-n-1],o[t-n-1],a[t-n-1].....p[t-2],v[t-2],a[t-2],p[t-1],v[t-1],a[t-1],p[t],v[t])

#### reward
-1 , the same as the gym environment 

In the following report , I will show DQN's performance under differnet learning rate and history number n.

lr : learning rate
n : history number 

I tried lr=0.001,0.0001 and n = 1,2
4 agents was trained and evaluated.





#### Analyze
I trained lots of network but all fail to reach the goal on the right mountain in 200 steps (training with 1000 episodes , max step 200 each epsiode).


It is a hard problem with  200 max steps.It seldom success reach the goal.Because the agent always got reward -1 , which makes agent(Q-nework) hard to learn how to success,it makes no difference to all actions.They are as worse as other action because they all get -1 rewards on whole trajectory, so it took long time to train a success agent.

I train a n=1 , lr=0.001 agent with 10000 epsidoes,it succeed in the training procedure between 2000~2500 episode,but after 2500 it fails to solve the mountain car problem .Maybe DQN is not a stable learning algorithm.


### Reports

Please check **analyze.ipynb** to see experiments figures.

##### evaluation results
######  &emsp; loss:root mean square error between  r+gamma*Q(s',a') and Q(s,a)
&emsp;&emsp; loss of 4 trained agent get lower and lower.
######  &emsp; reward sum
&emsp;&emsp;most of the reward sum is -199 (the last step I forgot to add to reward sum)


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