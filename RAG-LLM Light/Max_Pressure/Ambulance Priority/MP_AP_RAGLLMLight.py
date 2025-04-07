# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import traci
import sys, subprocess, os
import inspect
from read_net import NetworkData
import time
import ast
from RAG import interactive_query
from openai import OpenAI

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
net = NetworkData('environment.net.xml')


import re

def process_model_suggestion(text):
    pattern = r'\[\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(?:\s*,\s*-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)*?)\s*\]'
    matches = re.findall(pattern, text)

    if matches:
        last_array_str = f'[{matches[-1].replace(" ", "")}]'
        try:
            last_array = ast.literal_eval(last_array_str)
            if isinstance(last_array, list):
                return np.array(last_array, dtype=np.float64)
        except (SyntaxError, ValueError):
            print("Parsing failed:", last_array_str)
            return None

    return None


def phase_lane_get_normal(net):
    phase_1,phase_2,phase_3,phase_4 = [],[],[],[]
    phase = []
    for i in net.lane_data:
        if net.lane_data[i]['edge'] == 'E2TL' or net.lane_data[i]['edge'] == 'W2TL':
            if net.lane_data[i]['movement'] == 's' or net.lane_data[i]['movement'] == 'rs':
                phase_1.append(i)
            else :
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

def pressure_cal(phase,net):
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
    return total_pressure/len(phase)

def phase_light_normal(phase_now):
    if phase_now == 0:
        return 'rrrrrGGGGrrrrrrGGGGr'
    if phase_now == 1:
        return 'rrrrrrrrrGrrrrrrrrrG'
    if phase_now == 2:
        return 'GGGGrrrrrrGGGGrrrrrr'
    if phase_now == 3:
        return 'rrrrGrrrrrrrrrGrrrrr'  
    
    
def pressure_fun(phase):
    pressure = []
    for j in phase:
        #print(j)
        pressure.append(pressure_cal(j,net))

    pressure = np.array(pressure)
    return pressure

def max_pressure_phase(pressure):

    phase_index = pressure.argmax()
    print(pressure,phase_index)
    return phase_index

def llm_suggestion(phase, traffic_event,traffic_language,input,format,strategy):
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
    The each value in the array represent\n'''+
    str(input)+'''
    respectively.
    <Background Information>
    [Traffic Event]\n'''+
    str(traffic_event)+'''
    [Traffic Language]\n'''+
    str(traffic_language)+'''
    [The strategy]\n'''+
    str(strategy)+'''
    Then output the updated array.
    <Output and explanation>
    Output the updated pressure array in the following format:\n'''+
    str(format)+'''
    Ensure that the format remains identical to the input array.
    <Instructions>
    - Let's think step by step
    - Do not execute tasks that [The tasks] did not mention.
    - You can only give answer after finishing the analysis.
    - '''+str(format)+''' MUST be the final sentence of your answer.'''},
        ],
        stream=False
    )

    #model_run_time = time.time() - model_start_time
    #print(f"call_large_model Runtime：{model_run_time:.2f}sec")
    print(response.choices[0].message.content)
    pressure_llm = process_model_suggestion(response.choices[0].message.content)
    print(pressure_llm)
    return pressure_llm

def max_pressure_controller_normal(phase_pressure_max,phase_now,current_green_time,current_time,phase):
    min_green = 16
    max_green = 45
    phase_extend_time = 5
    if phase_pressure_max == phase_now:
        phase_next = phase_now
    else:
        phase_next = phase_pressure_max
    if phase_now == phase_next and current_green_time <max_green:
        traci.trafficlight.setRedYellowGreenState('TL',phase_light_normal(phase_now))
        current_green_time = current_green_time + phase_extend_time
        next_switch_time = current_time + phase_extend_time -1
        
        print(current_time,'Phase Extending',next_switch_time,phase_now)
    elif phase_now == phase_next and current_green_time >=max_green:
        pressure = pressure_fun(phase)
        pressure_exclude = []
        pressure_exclude = pressure.copy()
        pressure_exclude[phase_next] = -np.inf
        phase_next = pressure_exclude.argmax()
        traci.trafficlight.setRedYellowGreenState('TL',phase_light_normal(phase_next))
        current_green_time = min_green -1
        next_switch_time = current_time + min_green
        print(current_time,'Forced Phase Switching',next_switch_time,phase_now)
    elif (phase_now != phase_next):
        traci.trafficlight.setRedYellowGreenState('TL',phase_light_normal(phase_next))
        current_green_time = min_green -1
        next_switch_time = current_time + min_green
        print(current_time,'Phase Switching',next_switch_time,phase_now)
        
    print(phase_next)
        
    return current_green_time , next_switch_time


def is_near_lane_position(vehicle_id, target_position, threshold=10.0):
    lane_position = traci.vehicle.getLanePosition(vehicle_id)
    return abs(lane_position - target_position) <= threshold


def main():
    current_green_time = 15
    next_switch_time = 200
    target_vehicle_id = "emergencyVehicle1"
    target_position_enter = 635.0
    target_position_exit = 65.0
    emergency_in = 0
    emergency_done = False
    phase_4 = phase_lane_get_normal(net)
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

            if target_vehicle_id in traci.vehicle.getIDList() and not emergency_done:
                # 检测进入区域
                if is_near_lane_position(target_vehicle_id, target_position_enter) and emergency_in == 0:
                    emergency_in = 1
                    print(f"Vehicle {target_vehicle_id} In，emergency_in = 1")

                # 检测离开区域
                elif is_near_lane_position(target_vehicle_id, target_position_exit) and emergency_in == 1:
                    emergency_in = 0
                    emergency_done = True
                    print(f"Vehicle {target_vehicle_id} Out，emergency_in = 0")

            if current_time > 3600:
                break
            else:
                if current_time == next_switch_time:
                    if emergency_in == 1:
                        pressure_llm = llm_suggestion(phase_4, traffic_event,traffic_language,input,format,strategy)
                        phase_pressure_max = max_pressure_phase(pressure_llm)
                    else:
                        phase_pressure_max = max_pressure_phase(pressure_fun(phase_4))

                    phase_now = traci.trafficlight.getPhase('TL')
                    current_time = traci.simulation.getTime()
                    current_green_time, next_switch_time = max_pressure_controller_normal(
                        phase_pressure_max, phase_now, current_green_time, current_time, phase_4
                    )
                else:
                    continue
    finally:
        traci.close()

if __name__=='__main__':
    main()
    