from glo_vars import GloVars
from collections import OrderedDict

traci = GloVars.traci

class Vehicle:
    def __init__(self, vehID):
        self.id = vehID
        self.type = traci.vehicle.getTypeID(self.id)
        self.now = traci.simulation.getTime()
        self.start_time = self.now
        self.finish_time = 0
        self.log = OrderedDict()
        self.final_log = None
    
    def update(self):
        self.updateTime()
        self.cur_lane = traci.vehicle.getLaneID(self.id)
        self.cur_speed = traci.vehicle.getSpeed(self.id)
        self.log[self.now] = {
            'speed': self.cur_speed,
            'CO2_emission': traci.vehicle.getCO2Emission(self.id),
            'CO_emission': traci.vehicle.getCOEmission(self.id),
            'fuel_consumption': traci.vehicle.getFuelConsumption(self.id),
            'waiting_time': traci.vehicle.getAccumulatedWaitingTime(self.id),
            'distance': traci.vehicle.getDistance(self.id)
        }

    def updateTime(self):
        self.now = traci.simulation.getTime()

    def isFinished(self):
        return self.finish_time != 0

    def finish(self):
        self.updateTime()
        self.finish_time = self.now
        self.logFinal(is_finish=True)

    def logFinal(self, is_finish=False):
        self.updateTime()
        avg_speed = 0
        total_CO2_emission = 0
        total_CO_emission = 0
        total_fuel_consumption = 0
        waiting_time = 0
        count = 0
        for k, v in self.log.items():
            count += 1
            avg_speed += v['speed']
            total_CO2_emission += v['CO2_emission']
            total_CO_emission += v['CO_emission']
            total_fuel_consumption += v['fuel_consumption']

        if count > 0:
            self.final_log = {
                'avg_speed_per_step': avg_speed/count,
                'CO2_emission': total_CO2_emission,
                'CO_emission': total_CO_emission,
                'fuel_consumption': total_fuel_consumption,
                'waiting_time': self.log[self.now - 1]['waiting_time'],
                'distance': self.log[self.now - 1]['distance'],
                'travel_time': self.now - self.start_time,
                'avg_speed': self.log[self.now - 1]['distance'] / (self.now - self.start_time),
                'finished': is_finish
            }
        else:
            self.final_log = {
                'avg_speed_per_step': 0,
                'CO2_emission': total_CO2_emission,
                'CO_emission': total_CO_emission,
                'fuel_consumption': total_fuel_consumption,
                'waiting_time': self.log[self.now - 1]['waiting_time'],
                'distance': self.log[self.now - 1]['distance'],
                'travel_time': self.now - self.start_time,
                'avg_speed': 0,
                'finished': is_finish
            }
