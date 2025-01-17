from lux.utils import direction_to
import sys
import numpy as np
import torch
import torch.nn as nn

model = torch.load('q_net.pt')

def select_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.size(-1))
    else:
        return np.array([torch.argmax(q_values).item(),0,0])

class Agent_test():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.model = model

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        #observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        #observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        #num_agents = 16
        q_networkss = self.model
        q_values = torch.zeros((16, 6))
        epsilon = 0.05
        #unit_positions = state_batch[2]

        #for agent_id, q_network in zip(range(num_agents), q_networkss):

        for agent_id in available_unit_ids:
            #obs = torch.tensor(np.array(state[agent_id]), dtype=torch.float32).unsqueeze(0)
            q_network = q_networkss[agent_id]
            obs_input = torch.tensor(unit_positions[agent_id], dtype=torch.float32)
            #print(obs_input)
            q_values[agent_id] = q_network(obs_input)
            actions[agent_id] = select_action(q_values[agent_id], epsilon)
        return np.array(actions)
