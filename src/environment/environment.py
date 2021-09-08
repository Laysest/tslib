from src.environment.traffic_light import TrafficLight
import time
import sumolib
from sumolib.miscutils import getFreeSocketPort
import numpy as np
import sys
from src.glo_vars import GloVars
import tensorflow as tf
from src.environment.Vehicle import Vehicle
import pickle
import os
import threading
import cityflow

traci = GloVars.traci

def replay_in_parallel(traffic_light):
    traffic_light.replay()

last_vehs_id = []

class Environment():
    def __init__(self, config):
        """
            constructor of Environment
            config:{
                net: file defining traffic network,
                veh_type: file defining vehicle types,
                route: file defining workloads,
                end: maximum time of simulation,
                gui: True/False to enable gui
            }
        """
        self.vehicles = {}
        self.controllers = []
        self.config = config
        self.traffic_lights = None
        self.log = {}
        self.episode = -1
        GloVars.step = 0

    def reset(self):
        GloVars.step = 0
        self.vehicles = {}
        
    def update(self):
        def getDepartedAndArrivedVehiclesIDList():
            if GloVars.config['simulator'] == 'SUMO':
                return traci.simulation.getDepartedIDList(), traci.simulation.getArrivedIDList()
            elif GloVars.config['simulator'] == 'CityFlow':
                vehs_id = GloVars.eng.get_vehicles()
                departed_list = [veh for veh in vehs_id if veh not in self.vehicles]
                arrived_list = [veh for veh in self.vehicles if ((not self.vehicles[veh].isFinished()) and (veh not in vehs_id))]
                return departed_list, arrived_list

        departed_list, arrived_list = getDepartedAndArrivedVehiclesIDList()
        for veh_id_ in departed_list:
            if veh_id_ not in self.vehicles:
                self.vehicles[veh_id_] = Vehicle(veh_id_)

        for veh_id_ in arrived_list:
            self.vehicles[veh_id_].finish()

        for veh_id_ in self.vehicles:
            if not self.vehicles[veh_id_].isFinished():
                self.vehicles[veh_id_].update()
                self.vehicles[veh_id_].logStep(self.episode)
        GloVars.step += 1
    
    @staticmethod
    def startSimulation():
        if GloVars.config['simulator'] == 'SUMO':
            if GloVars.config['gui']:
                sumo_cmd = ["/usr/bin/sumo-gui"]
            else:
                sumo_cmd = ["/usr/bin/sumo"]

            sumo_config = ["-c", "./src/traffic-sumo/network.sumocfg", '-n', './src/traffic-sumo/%s' % GloVars.config['net'], '-r', './src/traffic-sumo/%s' % GloVars.config['route'], 
                        "-a", "./src/traffic-sumo/%s" % GloVars.config['veh_type'], "-e", str(GloVars.config['end'])]
            sumo_cmd.extend(sumo_config)
            traci.start(sumo_cmd)
        elif GloVars.config['simulator'] == 'CityFlow':
            eng = cityflow.Engine(config_file=GloVars.config['config_file'], thread_num=1)
            GloVars.eng = eng

    @staticmethod
    def nextStepSimulation():
        if GloVars.config['simulator'] == 'SUMO':
            traci.simulationStep()
        elif GloVars.config['simulator'] == 'CityFlow':
            GloVars.eng.next_step()

    @staticmethod
    def resetSimulation():
        if GloVars.config['simulator'] == 'SUMO':
            if GloVars.config['gui']:
                sumo_cmd = ["/usr/bin/sumo-gui"]
            else:
                sumo_cmd = ["/usr/bin/sumo"]

            sumo_config = ["-c", "./src/traffic-sumo/network.sumocfg", '-n', './src/traffic-sumo/%s' % GloVars.config['net'], '-r', './src/traffic-sumo/%s' % GloVars.config['route'], 
                        "-a", "./src/traffic-sumo/%s" % GloVars.config['veh_type'], "-e", str(GloVars.config['end'])]
            sumo_cmd.extend(sumo_config)
            traci.start(sumo_cmd)
        elif GloVars.config['simulator'] == 'CityFlow':
            GloVars.eng.reset()

    def train(self):
        """
            To run a simulation based on the configurate
        """
        pretrain_ = True
        ### Pre-train: ------------------------
        if pretrain_:
            self.episode = -1
            Environment.startSimulation()
            self.traffic_lights = [TrafficLight(config=tl) for tl in self.config['traffic_lights']]
            while GloVars.step < self.config['end']:
                self.update()
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].update(is_train=False, pretrain=True)
                    # self.traffic_lights[i].logStep(self.episode)
                Environment.nextStepSimulation()

            self.close()
        #### ------------------------------------------
        
        for e in range(50):
            self.episode = e
            print("Episode: %d" % e)
            Environment.resetSimulation()

            # create traffic_lights just once
            if e == 0 and pretrain_ == False:
                self.traffic_lights = [TrafficLight(config=tl) for tl in self.config['traffic_lights']]
            else:
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].reset()
            count = 1
            while GloVars.step < self.config['end']:
                self.update()
                GloVars.step += 1
                threads = {}
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].update(is_train=True)
                    # self.traffic_lights[i].logStep(self.episode)
                    if count % GloVars.INTERVAL == 0:
                        threads[i] = threading.Thread(target=replay_in_parallel, args=(self.traffic_lights[i],))
                        threads[i].start()
                        # self.traffic_lights[i].replay()
                if count % GloVars.INTERVAL == 0:
                    for i in range(len(self.traffic_lights)):
                        threads[i].join()
                Environment.nextStepSimulation()
                count += 1
            self.close(ep=e)

            if e > 0 and e % 5 == 0:
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].saveModel(e)
            print("-------------------------")
            print("")

        for i in range(len(self.traffic_lights)):
            self.traffic_lights[i].saveModel(e)

    def run(self):
        self.episode = -100
        Environment.startSimulation()
        self.traffic_lights = [TrafficLight(config=tl) for tl in self.config['traffic_lights']]
        for tf in self.traffic_lights:
            tf.loadModel()
        while GloVars.step < self.config['end']:
            self.update()
            for i in range(len(self.traffic_lights)):
                self.traffic_lights[i].update(is_train=False)
                # self.traffic_lights[i].logStep(self.episode)
            Environment.nextStepSimulation()
            if GloVars.step % 100 == 0:
                print(GloVars.step)
        self.close()


    def close(self, ep=-1):
        """
            close simulation
        """
        if GloVars.config['simulator'] == 'SUMO':
            traci.close()
        self.evaluate(ep)
        self.reset()

    def evaluate(self, ep):
        """
           Log results of this episode
        """
        self.log[ep] = {}
        for veh_id_ in self.vehicles:
            if not self.vehicles[veh_id_].isFinished():
                self.vehicles[veh_id_].logFinal()

        veh_logs = [veh.final_log for veh in self.vehicles.values()]
        metrics = ['avg_speed_per_step', 'CO2_emission', 'CO_emission', 'fuel_consumption', 'waiting_time',\
                    'distance', 'travel_time', 'avg_speed']
        self.log[ep]['veh_logs'] = veh_logs

        print("\n")
        for metric in metrics:
            val = sum(d[metric] for d in veh_logs) / len(veh_logs)
            print("{0:20}:{1}".format(metric, val))
            self.log[ep][metric] = val
        
        # if not os.path.exists(self.config['log_folder']):
        #     os.makedirs(self.config['log_folder'])
        # pickle.dump(self.log, open('%s/log.pkl' % self.config['log_folder'], 'wb'))
    
    