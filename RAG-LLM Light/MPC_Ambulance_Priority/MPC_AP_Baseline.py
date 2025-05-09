# -*- coding: utf-8 -*-

import traci
import traci.constants as tc
import numpy as np
import itertools
import time

phase_lane = {
    1: ['E2TL_0', 'E2TL_1', 'E2TL_2', 'W2TL_0', 'W2TL_1', 'W2TL_2'],  # EW_S
    2: ['E2TL_3', 'W2TL_3'],                                             # EW_L
    3: ['N2TL_0', 'N2TL_1', 'N2TL_2', 'S2TL_0', 'S2TL_1', 'S2TL_2'],     # NS_S
    4: ['N2TL_3', 'S2TL_3']                                              # NS_L
}


lane_ids = phase_lane[1] + phase_lane[2] + phase_lane[3] + phase_lane[4]
lane_to_index = {lane: idx for idx, lane in enumerate(lane_ids)}
prev_lane_vehicle_ids = {}


# Configuration
p_green_list = [0.256, 0.291, 0.277, 0.176]
T_horizon = 3
time_step = 1
min_phase_duration = 15   
max_phase_duration = 45

def measure_inflow(lane_ids, prev_lane_vehicle_ids):
    u = []
    for lane in lane_ids:
        current_ids = set(traci.lane.getLastStepVehicleIDs(lane))
        prev_ids = prev_lane_vehicle_ids.get(lane, set())
        new_vehicles = current_ids - prev_ids
        inflow_count = len(new_vehicles)
        u.append(inflow_count)
        prev_lane_vehicle_ids[lane] = current_ids
    return u


def get_current_state(lane_ids):
    x = []
    for lane in lane_ids:
        n = traci.lane.getLastStepVehicleNumber(lane)
        x.append(float(n))
    return np.array(x)

def l2_l1_penalty(x, lambda_l2=0.01, lambda_l1=0.005):
    l2_term = np.sum((x - np.mean(x)) ** 2)
    l1_term = np.sum(np.abs(x - np.mean(x)))

    return lambda_l2 * l2_term + lambda_l1 * l1_term

def simulate_ctm_sequence(x0, candidate_seq, u, p_green_list_fun, time_step):
    x = x0.copy()
    cost = 0  
    print(p_green_list_fun)
    # 遍历每个相位，正确匹配 p_green
    for i, phase in enumerate(candidate_seq):
        p = np.zeros(len(x))
        phase_p_green = p_green_list_fun[phase - 1]

        print(f"Current Phase: {phase}, Corresponding p_green: {phase_p_green}")

        for lane, idx in lane_to_index.items():
            if lane in phase_lane[phase]:  
                p[idx] = phase_p_green
            else:
                p[idx] = 0.0

        # CTM: x_next = max(x + u - p * x, 0)
        for _ in range(time_step * 15):
            x = np.maximum(x + u - p * x, 0)
        cost += l2_l1_penalty(x)

    return cost

def mpc_cost_array(x0, u, p_green_list, T_horizon,current_phase, phase_timer,groups=4):

    seq_list = []
    cost_array = []
    for seq in itertools.product([1, 2, 3, 4], repeat=T_horizon):
        cost = simulate_ctm_sequence(x0, seq, u, p_green_list,time_step)
        
        seq_list.append(seq)
        cost_array.append(cost)

    n = len(seq_list)
    group_size = n // groups
    group_min_costs = []
    for i in range(groups):
        start_index = i * group_size
        end_index = (i + 1) * group_size if i < groups - 1 else n
        group_costs = cost_array[start_index:end_index]
        min_cost = min(group_costs)
        group_min_costs.append(min_cost)
    print(group_min_costs)
    return group_min_costs


def mpc_optimal_sequence(group_min_costs):

    best_seq = np.argmin(group_min_costs)
    return best_seq
def mpc_optimal_sequence_secondbest(group_min_costs):
    unique_group_min_costs = sorted(set(group_min_costs))
    if len(unique_group_min_costs) < 2:
        raise ValueError("None")
    return unique_group_min_costs[1]

def choose_next_phase(current_phase, suggested_phase, phase_timer, min_phase_duration, max_phase_duration, candidate_seq, time_step):

        if phase_timer < min_phase_duration:
            chosen_phase = current_phase

        elif phase_timer < max_phase_duration:
            if suggested_phase == current_phase:
                chosen_phase = current_phase
            else:
                chosen_phase = suggested_phase
                phase_timer = 0

        else:
            if suggested_phase == current_phase:
                second_best_phase = mpc_optimal_sequence_secondbest(candidate_seq)
                chosen_phase = second_best_phase
                phase_timer = 0
            else:
                chosen_phase = suggested_phase
                phase_timer = 0

        return chosen_phase, phase_timer

def phase_light(phase_now):
    if phase_now == 0:
        return 'rrrrrGGGGrrrrrrGGGGr'
    if phase_now == 1:
        return 'rrrrrrrrrGrrrrrrrrrG'
    if phase_now == 2:
        return 'GGGGrrrrrrGGGGrrrrrr'
    if phase_now == 3:
        return 'rrrrGrrrrrrrrrGrrrrr'

# 下发交通灯控制命令至SUMO
def set_traffic_light(phase):
    traci.trafficlight.setRedYellowGreenState('TL',phase_light(phase))
    
def main():

    sumoCmd = ["sumo-gui", "-c", "sumo_config.sumocfg"]
    traci.start(sumoCmd)
    current_phase = 1
    phase_timer = 0

    try:
        for lane in lane_ids:
            prev_lane_vehicle_ids[lane] = set(traci.lane.getLastStepVehicleIDs(lane))
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            phase_timer += 1
            current_time = traci.simulation.getTime()
            print(phase_timer)
            x_current = get_current_state(lane_ids)
            u = measure_inflow(lane_ids, prev_lane_vehicle_ids)
            if phase_timer < min_phase_duration    :
                continue
            else :               

                group_min_costs = mpc_cost_array(x_current, u, p_green_list, T_horizon, current_phase, phase_timer)
                suggested_phase = mpc_optimal_sequence(group_min_costs)
                print(suggested_phase)

                chosen_phase, phase_timer = choose_next_phase(
                    current_phase, suggested_phase, phase_timer, 
                    min_phase_duration, max_phase_duration, group_min_costs, time_step
                )

                set_traffic_light(chosen_phase)
                print(chosen_phase)
                current_phase = chosen_phase

            if current_time >= 3600:
                break
                
    finally:
        traci.close()


if __name__ == "__main__":
    main()
