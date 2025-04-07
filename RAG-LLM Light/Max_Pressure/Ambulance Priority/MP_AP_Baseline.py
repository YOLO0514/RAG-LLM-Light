# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 22:50:59 2025

@author: YOLO
"""

import numpy as np
import pandas as pd
import traci
import sys, subprocess, os
import inspect
from read_net import NetworkData
import time


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
net = NetworkData('environment.net.xml')

def phase_lane_get(net):
    phase_1,phase_2,phase_3,phase_4 = [],[],[],[]
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

def phase_light(phase_now):
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

def max_pressure_phase(phase):
    pressure = pressure_fun(phase)

    phase_index = pressure.argmax()
    print(pressure,phase_index)
    return phase_index

def max_pressure_controller(phase_pressure_max,phase_now,current_green_time,current_time,phase):
    min_green = 16
    max_green = 45
    phase_extend_time = 5
    if phase_pressure_max == phase_now:
        phase_next = phase_now
    else:
        phase_next = phase_pressure_max
    
    if phase_now == phase_next and current_green_time <max_green:
        traci.trafficlight.setRedYellowGreenState('TL',phase_light(phase_now))
        current_green_time = current_green_time + phase_extend_time
        next_switch_time = current_time + phase_extend_time -1
        
        print(current_time,'Phase Extending',next_switch_time,phase_now)
    elif phase_now == phase_next and current_green_time >=max_green:
        pressure = pressure_fun(phase)
        pressure_exclude = []
        pressure_exclude = pressure.copy()
        pressure_exclude[phase_next] = -np.inf
        phase_next = pressure_exclude.argmax()
        traci.trafficlight.setRedYellowGreenState('TL',phase_light(phase_next))
        current_green_time = min_green -1
        next_switch_time = current_time + min_green
        print(current_time,'Forced Phase Switching',next_switch_time,phase_now)
    elif (phase_now != phase_next):
        traci.trafficlight.setRedYellowGreenState('TL',phase_light(phase_next))
        current_green_time = min_green -1
        next_switch_time = current_time + min_green
        print(current_time,'Phase Switching',next_swith_time,phase_now)
        
    print(phase_next)
        
    return current_green_time , next_swith_time



def main():
    current_green_time = 15
    next_switch_time = 200
    phase = phase_lane_get(net)
    traci.start(["sumo-gui", "-c", "sumo_config.sumocfg"])
    try :
        for step in range(3600):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
           
            if current_time >3600:
                break
            else:
                if current_time == next_switch_time:
                    phase_pressure_max = max_pressure_phase(phase)
                    phase_now = traci.trafficlight.getPhase('TL')
                    current_time = traci.simulation.getTime()
                    current_green_time,next_switch_time = max_pressure_controller(phase_pressure_max,phase_now,current_green_time,current_time,phase)
                else:
                    continue
    finally :
        traci.close()
            
if __name__=='__main__':
    main()
    
    
