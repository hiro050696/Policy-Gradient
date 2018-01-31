import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

gamma = 0.99

env = gym.make('CartPole-v0')
env.seed(543)
torch.manual_seed(543)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4,128)
        self.affine2 = nn.Linear(128,2)

        self.saved_log_probs = []
        self.rewards = []
    def forward(self,x):
        x = F.relu(self.affine1(x))
        x = F.softmax(self.affine2(x),dim=1)
        return x

policy = Policy()
optimizer = optim.Adam(policy.parameters(),lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = np.random.choice([0,1],p=probs.data[0].numpy())
    policy.saved_log_probs.append(torch.log(probs[0][action]))
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

        
def main():
    for i_episode in count(1):
        if i_episode % 10 == 0: print(i_episode)
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state,reward,done,_ = env.step(action)
            env.render()
            policy.rewards.append(reward)
            if done:
                break
        finish_episode()
    
    
if __name__ == '__main__':
    main()
