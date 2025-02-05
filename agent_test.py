from lux.utils import direction_to
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gen_data import gen_data

# 에이전트 별 q-network

class AgentQNetwork(nn.Module):
    def __init__(self, obs_shape=(5, 24, 24), action_space=(6, 24, 24)):
        super(AgentQNetwork, self).__init__()
        self.action_space = action_space
        self.conv1 = nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.LazyLinear(action_space[0] * action_space[1] * action_space[2])

    def forward(self, obs): # Input dimension -> (bs, 5, 24, 24)
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0) # 1개 짜리 input을 받은 경우 (1,5,24,24)로 변환
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)  # Flatten for the fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x).view((obs.shape)[0], *self.action_space) 
        return q_values # Out dimension -> (bs, 6, 24, 24).

class Agent_test():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.model = torch.load('q_net.pt')

    def optimal_action_from_qval(self, q_value): # sinlge agent의 batched q value(bs, 6,24,24)를 list로 받음
        batch_size = len(q_value)
        max_q, max_id = torch.max(q_value.view(batch_size, -1), dim =1)
        unraveled_idx = torch.tensor([torch.unravel_index(idx, q_value.shape[1:]) for idx in max_id])
        q_values_batched = [max_q,unraveled_idx]
        return q_values_batched # output -> list, q_values_batched = [(bs,1), (bs,3)], 첫 째는 optimal q value, 둘 째는 해당하는 action 자체

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs[self.player]["units_mask"][self.team_id]) # shape (max_units, )
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        #num_agents = 16
        q_networks = self.model
        epsilon = 0.05

        num_agent = 16 
        imaged_obs = torch.tensor(gen_data(obs, True)[self.player], dtype=torch.float)

        with torch.no_grad():
            for agent_id ,q_network in zip(range(num_agent), q_networks):
                q_network.eval()
                q_obs = q_network(imaged_obs)
                optimal_q = self.optimal_action_from_qval(q_obs)
                if agent_id in available_unit_ids:
                    if np.random.rand() < epsilon:
                        rand_act = torch.tensor([np.random.randint(6),np.random.randint(24),np.random.randint(24)])
                        rand_act = rand_act.repeat(q_obs.size(0),1)
                        actions[agent_id]= rand_act
                    else:
                        actions[agent_id] = optimal_q[1]
        return np.array(actions) # action을 담은 list
