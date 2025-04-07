# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import traci
import sys, subprocess, os
import inspect
from read_net import NetworkData
import time
import ast
from openai import OpenAI
from RAG import interactive_query

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

net = NetworkData('environment.net.xml')

import re


def process_model_suggestion(text):
    pattern = r'\[\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(?:\.\.\.)?(?:\s*,\s*-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(?:\.\.\.)?)*?)\s*\]'
    matches = re.findall(pattern, text)
    if matches:
        cleaned_match = re.sub(r"(\d+\.\d*)\.\.\.", r"\1", matches[-1])
        last_array_str = "[" + cleaned_match.replace(" ", "") + "]"

        try:
            last_array = ast.literal_eval(last_array_str)
            if isinstance(last_array, list):
                return np.array(last_array, dtype=np.float64)
        except (SyntaxError, ValueError):
            print("Parsing failed:", last_array_str)
            return None

    return None


def phase_lane_get_llm(net):
    phase_1, phase_2, phase_3, phase_4, phase_5 = [], [], [], [], []
    phase = []
    for i in net.lane_data:
        if net.lane_data[i]['edge'] == 'E2TL':
            if net.lane_data[i]['movement'] == 's' or net.lane_data[i]['movement'] == 'rs':
                phase_1.append(i)
        if net.lane_data[i]['edge'] == 'W2TL':
            if net.lane_data[i]['movement'] == 's' or net.lane_data[i]['movement'] == 'rs':
                phase_2.append(i)
        if net.lane_data[i]['edge'] == 'E2TL' or net.lane_data[i]['edge'] == 'W2TL':
            if net.lane_data[i]['movement'] == 'l':
                phase_3.append(i)
        if net.lane_data[i]['edge'] == 'N2TL' or net.lane_data[i]['edge'] == 'S2TL':
            if net.lane_data[i]['movement'] == 's' or net.lane_data[i]['movement'] == 'rs':
                phase_4.append(i)
            elif net.lane_data[i]['movement'] == 'l':
                phase_5.append(i)
            else:
                continue
    phase.append(phase_1)
    phase.append(phase_2)
    phase.append(phase_3)
    phase.append(phase_4)
    phase.append(phase_5)
    print(phase)

    return phase


# [e_rs][w_rs][ew_l][ns_rs][ns_l]

def phase_lane_get_normal(net):
    phase_1, phase_2, phase_3, phase_4 = [], [], [], []
    phase = []
    for i in net.lane_data:
        if net.lane_data[i]['edge'] == 'E2TL' or net.lane_data[i]['edge'] == 'W2TL':
            if net.lane_data[i]['movement'] == 's' or net.lane_data[i]['movement'] == 'rs':
                phase_1.append(i)
            else:
                phase_2.append(i)

        else:
            if net.lane_data[i]['movement'] == 's' or net.lane_data[i]['movement'] == 'rs':
                phase_3.append(i)
            elif net.lane_data[i]['movement'] == 'l':
                phase_4.append(i)
            else:
                continue

    phase.append(phase_1)
    phase.append(phase_2)
    phase.append(phase_3)
    phase.append(phase_4)

    return phase


# [ew_rs][ew_l][ns_rs][ns_l]

def pressure_cal(phase, net):
    total_pressure = 0
    income_pressure = 0
    outcome_pressure = 0
    for income_lane in phase:
        lane_length_income = traci.lane.getLength(income_lane)
        lane_length_outcome = traci.lane.getLength(next(iter(net.lane_data[income_lane]['outgoing'])))
        vehicle_ids_income = traci.lane.getLastStepVehicleIDs(income_lane)
        vehicle_ids_outcome = traci.lane.getLastStepVehicleIDs(next(iter(net.lane_data[income_lane]['outgoing'])))
        for vehicle_id in vehicle_ids_income:
            position = traci.vehicle.getLanePosition(vehicle_id)
            if lane_length_income - position <= 750:
                income_pressure += 1
        for vehicle_id in vehicle_ids_outcome:
            position = traci.vehicle.getLanePosition(vehicle_id)
            if position - lane_length_outcome <= 750:
                outcome_pressure += 1
        total_pressure += income_pressure - outcome_pressure
    return total_pressure / len(phase)


def car_removing(phase, net, time_threshold, dist_threshold, current_time, stay_time, excluded_vehicles):
    for lane_group in phase:
        for income_lane in lane_group:
            lane_length_income = traci.lane.getLength(income_lane)
            vehicle_ids_income = traci.lane.getLastStepVehicleIDs(income_lane)
            for vehicle_id in vehicle_ids_income:
                if vehicle_id in excluded_vehicles:
                    continue
                position = traci.vehicle.getLanePosition(vehicle_id)
                if lane_length_income - position > dist_threshold:
                    if vehicle_id not in stay_time:
                        stay_time[vehicle_id] = 0
                    stay_time[vehicle_id] += 1
                else:
                    stay_time[vehicle_id] = 0

                if stay_time.get(vehicle_id, 0) >= time_threshold:
                    print(f"Removing vehicle {vehicle_id} at time {current_time:.1f} s")
                    traci.vehicle.remove(vehicle_id)
                    del stay_time[vehicle_id]
    return stay_time


def is_incorrect_lane(vehID):
    route = traci.vehicle.getRoute(vehID)
    if len(route) < 2:
        return False
    next_edge = route[1]
    currentLaneID = traci.vehicle.getLaneID(vehID)
    current_edge = currentLaneID.split('_')[0]
    if current_edge == next_edge:
        return False
    links = traci.lane.getLinks(currentLaneID)
    possible_next_edges = {link[0].split('_')[0] for link in links}
    return next_edge not in possible_next_edges


def check_and_remove_stuck_vehicles(waiting_threshold, phase, current_time):
    for lane_group in phase:
        for income_lane in lane_group:
            # Only deal with this lane for the purpose of improving simulation efficiency.
            if income_lane != "W2TL_3":
                continue
            lane_length_income = traci.lane.getLength(income_lane)
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(income_lane)
            for vehID in vehicles_on_lane:
                position = traci.vehicle.getLanePosition(vehID)
                if lane_length_income - position > 30:
                    continue
                if not is_incorrect_lane(vehID):
                    continue
                waiting_time = traci.vehicle.getWaitingTime(vehID)
                if waiting_time > waiting_threshold:
                    print(f"At time: {current_time} sec, Remove vehicle {vehID}, Waiting time: {waiting_time} sec")
                    traci.vehicle.remove(vehID)


def phase_light_normal(phase_now):
    if phase_now == 0:
        return 'rrrrrGGGGrrrrrrGGGGr'
    if phase_now == 1:
        return 'rrrrrrrrrGrrrrrrrrrG'
    if phase_now == 2:
        return 'GGGGrrrrrrGGGGrrrrrr'
    if phase_now == 3:
        return 'rrrrGrrrrrrrrrGrrrrr'


def phase_light_llm(phase_now):
    # Phase Truncation
    if phase_now == 0:
        return 'rrrrrGGGGrrrrrrGGGGr'
    if phase_now == 1:
        return 'rrrrrGGGGrrrrrrGGGGr'
    if phase_now == 2:
        return 'rrrrrrrrrGrrrrrrrrrG'
    if phase_now == 3:
        return 'GGGGrrrrrrGGGGrrrrrr'
    if phase_now == 4:
        return 'rrrrGrrrrrrrrrGrrrrr'


def pressure_fun(phase):
    pressure = []
    for j in phase:
        # print(j)
        pressure.append(pressure_cal(j, net))

    pressure = np.array(pressure)
    return pressure


def max_pressure_phase(pressure):
    phase_index = pressure.argmax()
    print(pressure, phase_index)
    return phase_index


def llm_suggestion(phase, traffic_event, traffic_language, input, format, strategy):
    #model_start_time = time.time()
    pressure = pressure_fun(phase)
    print(pressure)

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are now a tuning agent for control algorithm parameters."},
            {"role": "user", "content": '''
    <Input and explanation>
    You are now a tuning agent for Max-Pressure control algorithm parameters.
    The pressure for each phase is given below:''' + str(pressure) + ''' .
    The each value in the array represent\n''' +
    str(input) + '''
    respectively.
    <Background Information>
    [Traffic Event]\n''' +
    str(traffic_event) + '''
    [The strategy]
    Depending on the [Traffic Event].
    Then output the updated array.
    <Output and explanation>
    Output the updated pressure array in the following format:\n''' +
    str(format) + '''
    Ensure that the format remains identical to the input array.
    <Instructions>
    - Let's think step by step
    - Do not execute tasks that [The tasks] did not mention.
    - You can only give answer after finishing the analysis.
    - ''' + str(format) + ''' MUST be the final sentence of your answer.'''},
        ],
        stream=False
    )
    # model_run_time = time.time() - model_start_time
    # print(f"call_large_model Runtime：{model_run_time:.2f}sec")
    print(response.choices[0].message.content)
    pressure_llm = process_model_suggestion(response.choices[0].message.content)
    print(pressure_llm)
    return pressure_llm


def max_pressure_controller_llm(phase_pressure_max, phase_now, current_green_time, current_time, phase):
    min_green = 16
    max_green = 45
    phase_extend_time = 5
    if phase_pressure_max == phase_now:
        phase_next = phase_now
    else:
        phase_next = phase_pressure_max
    if phase_now == phase_next and current_green_time < max_green:
        traci.trafficlight.setRedYellowGreenState('TL', phase_light_llm(phase_now))
        current_green_time = current_green_time + phase_extend_time
        next_switch_time = current_time + phase_extend_time - 1

        print(current_time, 'Phase Extending', next_switch_time, phase_now)
    elif phase_now == phase_next and current_green_time >= max_green:
        pressure = pressure_fun(phase)
        pressure_exclude = []
        pressure_exclude = pressure.copy()
        pressure_exclude[1] = -np.inf
        pressure_exclude[phase_next] = -np.inf
        phase_next = pressure_exclude.argmax()
        traci.trafficlight.setRedYellowGreenState('TL', phase_light_llm(phase_next))
        current_green_time = min_green - 1
        next_switch_time = current_time + min_green
        print(current_time, 'Forced Phase Switching', next_switch_time, phase_now)
    elif (phase_now != phase_next):

        traci.trafficlight.setRedYellowGreenState('TL', phase_light_llm(phase_next))
        current_green_time = min_green - 1
        next_switch_time = current_time + min_green
        print(current_time, 'Phase Switching', next_switch_time, phase_now)

    print(phase_next)

    return current_green_time, next_switch_time


def max_pressure_controller_normal(phase_pressure_max, phase_now, current_green_time, current_time, phase):
    min_green = 16
    max_green = 45
    phase_extend_time = 5
    if phase_pressure_max == phase_now:
        phase_next = phase_now
    else:
        phase_next = phase_pressure_max
    if phase_now == phase_next and current_green_time < max_green:
        traci.trafficlight.setRedYellowGreenState('TL', phase_light_normal(phase_now))
        current_green_time = current_green_time + phase_extend_time
        next_switch_time = current_time + phase_extend_time - 1

        print(current_time, 'Phase Extending', next_switch_time, phase_now)
    elif phase_now == phase_next and current_green_time >= max_green:
        pressure = pressure_fun(phase)
        pressure_exclude = []
        pressure_exclude = pressure.copy()
        pressure_exclude[phase_next] = -np.inf
        phase_next = pressure_exclude.argmax()
        traci.trafficlight.setRedYellowGreenState('TL', phase_light_normal(phase_next))
        current_green_time = min_green - 1
        next_switch_time = current_time + min_green
        print(current_time, 'Forced Phase Switching', next_switch_time, phase_now)
    elif (phase_now != phase_next):
        traci.trafficlight.setRedYellowGreenState('TL', phase_light_normal(phase_next))
        current_green_time = min_green - 1
        next_switch_time = current_time + min_green
        print(current_time, 'Phase Switching', next_switch_time, phase_now)

    print(phase_next)

    return current_green_time, next_switch_time


def main():
    current_green_time = 15
    next_swith_time = 200
    is_stop = 0
    all_stop = 0
    max_stoptime = 0
    stay_time = {}
    time_threshold, dist_threshold = 180, 150
    waiting_threshold = 15
    duration = 1500
    pos = 740
    s_v = {
        "v_1": {
            "vehID": "1.0",
            "edgeID": "W2TL",
            "pos": pos,
            "laneIndex": 2,
            "duration": duration
        },
        "v_2": {
            "vehID": "1.1",
            "edgeID": "W2TL",
            "pos": pos,
            "laneIndex": 1,
            "duration": duration
        },
        "v_3": {
            "vehID": "1.2",
            "edgeID": "W2TL",
            "pos": pos - 10,
            "laneIndex": 2,
            "duration": duration
        },
        "v_4": {
            "vehID": "1.3",
            "edgeID": "W2TL",
            "pos": pos - 10,
            "laneIndex": 1,
            "duration": duration
        },
    }
    corre_veh = s_v["v_1"]
    phase_5 = phase_lane_get_llm(net)
    phase_4 = phase_lane_get_normal(net)
    excluded_vehicles = {s_v["v_2"]["vehID"], s_v["v_3"]["vehID"], s_v["v_4"]["vehID"], s_v["v_1"]["vehID"]}
    rag_result = []
    rag_result = interactive_query()
    traffic_event = rag_result['event_description']
    traffic_language = rag_result['standard_language']
    input = rag_result['input_condition']
    format = rag_result['format_requirement']
    strategy = rag_result['strategy']
    traci.start(["sumo-gui", "-c", "sumo_config.sumocfg"])
    try:
        for step in range(3600):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            if all_stop == 0:
                if current_time == 4.0:
                    traci.vehicle.setStop(s_v["v_2"]["vehID"], s_v["v_2"]["edgeID"], pos=s_v["v_2"]["pos"],
                                          laneIndex=s_v["v_2"]["laneIndex"], duration=s_v["v_2"]["duration"], flags=0)
                if current_time == 7.0:
                    traci.vehicle.setStop(s_v["v_3"]["vehID"], s_v["v_3"]["edgeID"], pos=s_v["v_3"]["pos"],
                                          laneIndex=s_v["v_3"]["laneIndex"], duration=s_v["v_3"]["duration"], flags=0)
                if current_time == 10.0:
                    traci.vehicle.setStop(s_v["v_4"]["vehID"], s_v["v_4"]["edgeID"], pos=s_v["v_4"]["pos"],
                                          laneIndex=s_v["v_4"]["laneIndex"], duration=s_v["v_4"]["duration"], flags=0)
                    all_stop = 1
                    max_stoptime = current_time
                    corre_veh = s_v["v_4"]
                else:
                    traci.vehicle.setStop(s_v["v_1"]["vehID"], s_v["v_1"]["edgeID"], pos=s_v["v_1"]["pos"],
                                          laneIndex=s_v["v_1"]["laneIndex"], duration=s_v["v_1"]["duration"], flags=0)
                    is_stop = 1
            if current_time >= max_stoptime + corre_veh["duration"]:
                is_stop = 0
            if current_time < 200:
                continue
            elif current_time > 3600:
                break
            else:
                if current_time == next_swith_time:
                    if is_stop == 0:
                        check_and_remove_stuck_vehicles(waiting_threshold, phase_4, current_time)
                        stay_time = car_removing(phase_4, net, time_threshold, dist_threshold, current_time, stay_time,
                                                 excluded_vehicles)
                        phase_pressure_max = max_pressure_phase(pressure_fun(phase_4))
                        phase_now = traci.trafficlight.getPhase('TL')
                        current_time = traci.simulation.getTime()
                        current_green_time, next_switch_time = max_pressure_controller_normal(phase_pressure_max,
                                                                                              phase_now,
                                                                                              current_green_time,
                                                                                              current_time, phase_4)
                    else:
                        check_and_remove_stuck_vehicles(waiting_threshold, phase_5, current_time)
                        stay_time = car_removing(phase_5, net, time_threshold, dist_threshold, current_time, stay_time, excluded_vehicles)
                        pressure_llm = llm_suggestion(phase_5, traffic_event, traffic_language, input, format,strategy)
                        phase_pressure_max = max_pressure_phase(pressure_llm)
                        phase_now = traci.trafficlight.getPhase('TL')
                        current_time = traci.simulation.getTime()
                        current_green_time, next_switch_time = max_pressure_controller_llm(phase_pressure_max,
                                                                                             phase_now,
                                                                                             current_green_time,
                                                                                             current_time, phase_5)
                else:
                    continue
    finally:
        traci.close()


if __name__ == '__main__':
    main()
