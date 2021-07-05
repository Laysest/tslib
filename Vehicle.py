from itertools import count
from numpy.core.numeric import full
from glo_vars import GloVars
from collections import OrderedDict
import pandas as pd
import os

traci = GloVars.traci

class Vehicle:
    def __init__(self, vehID):
        self.id = vehID
        self.now = GloVars.step
        self.start_time = self.now
        self.finish_time = 0
        if GloVars.config['simulator'] == 'SUMO':
            self.type = traci.vehicle.getTypeID(self.id)
            self.route = traci.vehicle.getRoute(self.id)
            lane_ = traci.vehicle.getLaneID(self.id)
            self.status = {
                'lane': lane_,
                'speed': traci.vehicle.getSpeed(self.id),
                'CO2_emission': traci.vehicle.getCO2Emission(self.id),
                'CO_emission': traci.vehicle.getCOEmission(self.id),
                'fuel_consumption': traci.vehicle.getFuelConsumption(self.id),
                'waiting_time': traci.vehicle.getAccumulatedWaitingTime(self.id),
                'distance': traci.vehicle.getDistance(self.id),
                'route': [{
                            'edge': lane_[:-2],
                            'first_time_step': self.now,
                            'last_time_step': self.now
                }]
            }
        elif GloVars.config['simulator'] == 'CityFlow':
            self.type = 'Nan'
            veh = GloVars.eng.get_vehicle_info(vehID)
            self.route = veh['route'][:-1].split(" ")
            lane_ = veh['drivable']
            self.status = {
                'lane': lane_,
                'speed': float(veh['speed']),
                'CO2_emission': 0,
                'CO_emission': 0,
                'fuel_consumption': 0,
                'waiting_time': 0,
                'distance': 0,
                'route': [{
                            'edge': lane_[:-2],
                            'first_time_step': self.now,
                            'last_time_step': self.now
                }]
            }

        self.log = {
            'total_speed': 0,
            'total_CO2_emission': 0,
            'total_CO_emission': 0,
            'total_fuel_consumption': 0,
            'count': 0
        }

    def update(self):
        self.updateTime()
        if GloVars.config['simulator'] == 'SUMO':
            lane = traci.vehicle.getLaneID(self.id)
            speed = traci.vehicle.getSpeed(self.id)
            CO2_emission = traci.vehicle.getCO2Emission(self.id)
            CO_emission = traci.vehicle.getCOEmission(self.id)
            fuel_consumption = traci.vehicle.getFuelConsumption(self.id)
            waiting_time = traci.vehicle.getAccumulatedWaitingTime(self.id)
            distance = traci.vehicle.getDistance(self.id)
        elif GloVars.config['simulator'] == 'CityFlow':
            veh_ = GloVars.eng.get_vehicle_info(self.id)
            lane = veh_['drivable']
            speed = float(veh_['speed'])
            CO2_emission = 0
            CO_emission = 0
            fuel_consumption = 0
            distance = 0
            if speed >= 0.1:
                waiting_time = 0
            else:
                waiting_time = self.status['waiting_time'] + 1

        last_route = self.status['route'].copy()
        edge = lane[:-2]
        route = {}

        # still current edge
        if edge == last_route[-1]['edge']:
            last_route[-1]['last_time_step'] = self.now
            route = last_route
        # next edge
        else:
            last_route.append({
                'edge': edge,
                'first_time_step': self.now,
                'last_time_step': self.now
            })
            route = last_route

        self.status = {
            'lane': lane,
            'speed': speed,
            'CO2_emission': CO2_emission,
            'CO_emission': CO2_emission,
            'fuel_consumption': fuel_consumption,
            'waiting_time': waiting_time,
            'distance': distance,
            'route': route
        }

        self.log['count'] += 1
        self.log['total_speed'] += speed
        self.log['total_CO2_emission'] += CO2_emission
        self.log['total_CO_emission'] += CO_emission
        self.log['total_fuel_consumption'] += fuel_consumption

    def logStep(self, episode):
        self.updateTime()
        self.status['episode'] = episode
        self.status['step'] = self.now
        df = pd.DataFrame([self.status])

        if not os.path.exists(GloVars.config['log_folder']):
            os.makedirs(GloVars.config['log_folder'])
        log_folder = '%s/log_veh_per_step.csv' % GloVars.config['log_folder']
        if not os.path.isfile(log_folder):
            df.to_csv(log_folder, header='column_names', index=False)
        else: # else it exists so append without writing the header
            df.to_csv(log_folder, mode='a', header=False, index=False)

    def updateTime(self):
        self.now = GloVars.step

    def isFinished(self):
        return self.finish_time != 0

    def finish(self):
        self.updateTime()
        self.finish_time = self.now
        self.logFinal(is_finish=True)

    def logFinal(self, is_finish=False):
        self.updateTime()

        if self.log['count'] > 0:
            self.final_log = {
                'id': self.id,
                'route': self.route,
                'type': self.type,
                'start': self.start_time,
                'avg_speed_per_step': self.log['total_speed']/self.log['count'],
                'CO2_emission': self.log['total_CO2_emission'],
                'CO_emission': self.log['total_CO_emission'],
                'fuel_consumption': self.log['total_fuel_consumption'],
                'waiting_time': self.status['waiting_time'],
                'distance': self.status['distance'],
                'travel_time': self.now - self.start_time,
                'avg_speed': self.status['distance'] / (self.now - self.start_time),
                'finished': is_finish,
                'route_detail': self.status['route']
            }
        else:
            self.final_log = {
                'id': self.id,
                'route': self.route,
                'type': self.type,
                'avg_speed_per_step': 0,
                'CO2_emission': 0,
                'CO_emission': 0,
                'fuel_consumption': 0,
                'waiting_time': 0,
                'distance': 0,
                'travel_time': 0,
                'avg_speed': 0,
                'finished': is_finish,
                'route_detail': self.status['route']
            }