# -*- coding: utf-8 -*-

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
    print(phase)
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

def car_removing(phase, net, time_threshold, dist_threshold, current_time,stay_time,excluded_vehicles):
    for lane_group in phase:
        for income_lane in lane_group:
            lane_length_income = traci.lane.getLength(income_lane)
            vehicle_ids_income = traci.lane.getLastStepVehicleIDs(income_lane)
            for vehicle_id in vehicle_ids_income:
                if vehicle_id in excluded_vehicles:
                    continue
                position = traci.vehicle.getLanePosition(vehicle_id)
                # 计算车辆距离交叉口的距离
                if lane_length_income - position > dist_threshold:  
                    # 若车辆不在字典中，先初始化为0
                    if vehicle_id not in stay_time:
                        stay_time[vehicle_id] = 0
                    stay_time[vehicle_id] += 1
                else:
                    stay_time[vehicle_id] = 0
                if stay_time.get(vehicle_id, 0) >= time_threshold:
                    print(f"Removing vehicle {vehicle_id} at time {current_time:.1f} s")
                    traci.vehicle.remove(vehicle_id)
                    del stay_time[vehicle_id]
    #print(stay_time)
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

def check_and_remove_stuck_vehicles(waiting_threshold, phase,current_time):

    for lane_group in phase:
        for income_lane in lane_group:
            # Only deal with this lane.
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
        print(current_time,'Phase Switching',next_switch_time,phase_now)
        
    print(phase_next)
        
    return current_green_time , next_switch_time



def main():
    current_green_time = 15
    next_switch_time = 200
    is_stop=0
    all_stop=0
    max_stoptime=0
    stay_time = {}
    time_threshold, dist_threshold = 180 , 150
    waiting_threshold = 15
    duration = 1500
    pos =740
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
        "pos": pos-10,
        "laneIndex": 2,
        "duration": duration
    },
    "v_4": {
        "vehID": "1.3",
        "edgeID": "W2TL",
        "pos": pos-10,
        "laneIndex": 1,
        "duration": duration
    },
}
    corre_veh = s_v["v_1"]
    phase = phase_lane_get(net)
    excluded_vehicles = {s_v["v_2"]["vehID"], s_v["v_3"]["vehID"], s_v["v_4"]["vehID"],s_v["v_1"]["vehID"]}
    traci.start(["sumo-gui", "-c", "sumo_config.sumocfg"])
    try :
        for step in range(3600):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            if all_stop==0 :
                if current_time == 4.0 :
                    traci.vehicle.setStop(s_v["v_2"]["vehID"],s_v["v_2"]["edgeID"],pos=s_v["v_2"]["pos"],laneIndex=s_v["v_2"]["laneIndex"],duration=s_v["v_2"]["duration"],flags=0)
                if current_time == 12.0 :
                    traci.vehicle.setStop(s_v["v_3"]["vehID"],s_v["v_3"]["edgeID"],pos=s_v["v_3"]["pos"],laneIndex=s_v["v_3"]["laneIndex"],duration=s_v["v_3"]["duration"],flags=0)
                if current_time == 16.0 :
                    traci.vehicle.setStop(s_v["v_4"]["vehID"],s_v["v_4"]["edgeID"],pos=s_v["v_4"]["pos"],laneIndex=s_v["v_4"]["laneIndex"],duration=s_v["v_4"]["duration"],flags=0)
                    all_stop = 1
                    max_stoptime = current_time
                    corre_veh = s_v["v_4"]
                else :
                    traci.vehicle.setStop(s_v["v_1"]["vehID"],s_v["v_1"]["edgeID"],pos=s_v["v_1"]["pos"],laneIndex=s_v["v_1"]["laneIndex"],duration=s_v["v_1"]["duration"],flags=0)
                    is_stop=1
            if current_time >= max_stoptime + corre_veh["duration"] :
                is_stop=0
            if current_time < 200 :
                continue
            elif current_time >3600:
                break
            else:
                if current_time == next_switch_time:
                    check_and_remove_stuck_vehicles(waiting_threshold,phase,current_time)
                    stay_time = car_removing(phase, net, time_threshold, dist_threshold, current_time,stay_time,excluded_vehicles)
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
    
    
