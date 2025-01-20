# Observation -> (5, 24, 24) array

def gen_data(obs: dict) -> dict:
    my_data = dict()
    num_teams = 2
    width, height = 24, 24

    for t in range(num_teams):
        my_obs = obs[f'player_{t}']

        # unit_position and unit_energy
        unit_position = np.full((width, height), -1)
        unit_energy = np.full((width, height), -1)
        for idx, my_unit_mask in enumerate(my_obs.units_mask[0]):
            if my_unit_mask:
                position = my_obs.units.position[0][idx]
                energy = my_obs.units.energy[0][idx]

                position_x, position_y = position[0], position[1]

                unit_position[position_x][position_y] = 0
                unit_energy[position_x][position_y] = energy
                
        for idx, opponent_unit_mask in enumerate(my_obs.units_mask[1]):
            if opponent_unit_mask:
                position = my_obs.units.position[1][idx]
                energy = my_obs.units.energy[1][idx]

                position_x, position_y = position[0], position[1]

                unit_position[position_x][position_y] = 1
                unit_energy[position_x][position_y] = energy

        # map information
        tile_info = my_obs.map_features.tile_type
        tile_energy = my_obs.map_features.energy

        relic_position = np.full((width, height), -1)
        for idx, relic_mask in enumerate(my_obs.relic_nodes_mask):
            if relic_mask:
                position = my_obs.relic_nodes[idx]

                position_x, position_y = position[0], position[1]

                relic_position[position_x][position_y] = 1

        my_arr = np.stack((unit_position, unit_energy, tile_info, tile_energy, relic_position))

        my_data[f'player_{t}'] = my_arr

    return my_data
