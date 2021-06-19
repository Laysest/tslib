from traffic_light import TrafficLight
import time
import sumolib
from sumolib.miscutils import getFreeSocketPort
import numpy as np
import sys
from glo_vars import GloVars
import tensorflow as tf
from Vehicle import Vehicle
import pickle
import os
import threading

traci = GloVars.traci

def replay_in_parallel(traffic_light):
    traffic_light.replay()

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

    def reset(self):
        self.vehicles = {}
        
    def update(self):
        now = traci.simulation.getTime()

        for veh_id_ in traci.simulation.getDepartedIDList():
            if veh_id_ not in self.vehicles:
                self.vehicles[veh_id_] = Vehicle(veh_id_)

        for veh_id_ in traci.simulation.getArrivedIDList():
            self.vehicles[veh_id_].finish()

        for veh_id_ in self.vehicles:
            if not self.vehicles[veh_id_].isFinished():
                self.vehicles[veh_id_].update()
                self.vehicles[veh_id_].logStep(self.episode)
        GloVars.vehicles = self.vehicles
    
    def getEdgesOfNode(self, tf_id):
        in_edges = []
        out_edges = []
        for edge in self.edges:
            if edge.getToNode() == tf_id:
                in_edges.append(edge)
            # elif edge.get

    def train(self):
        """
            To run a simulation based on the configurate
        """
        if self.config["gui"]:
            sumo_cmd = ["/usr/bin/sumo-gui"]
        else:
            sumo_cmd = ["/usr/bin/sumo"]

        sumo_config = ["-c", "./traffic-sumo/network.sumocfg", '-n', './traffic-sumo/%s' % self.config['net'], '-r', './traffic-sumo/%s' % self.config['route'], 
                      "-a", "./traffic-sumo/%s" % self.config['veh_type'], "-e", str(self.config['end'])]
        sumo_cmd.extend(sumo_config)

        # self.edges = str(sumolib.net.readNet('./traffic-sumo/%s' % self.config['net']).getEdges()
        pretrain_ = True
        ### Pre-train: ------------------------
        if pretrain_:
            self.episode = -1
            traci.start(sumo_cmd)
            self.traffic_lights = [TrafficLight(config=tl) for tl in self.config['traffic_lights']]
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
                self.update()
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].update(is_train=False, pretrain=True)
                    self.traffic_lights[i].log_step()
                traci.simulationStep()
            self.close()
        #### ------------------------------------------

        for e in range(50):
            self.episode = e
            print("Episode: %d" % e)
            traci.start(sumo_cmd)

            # create traffic_lights just once
            if e == 0 and pretrain_ == False:
                self.traffic_lights = [TrafficLight(config=tl) for tl in self.config['traffic_lights']]
            else:
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].reset()
            count = 1
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
                self.update()
                GloVars.step += 1
                threads = {}
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].update(is_train=True)
                    self.traffic_lights[i].log_step()
                    if count % GloVars.INTERVAL == 0:
                        threads[i] = threading.Thread(target=replay_in_parallel, args=(self.traffic_lights[i],))
                        threads[i].start()
                        # self.traffic_lights[i].replay()
                if count % GloVars.INTERVAL == 0:
                    for i in range(len(self.traffic_lights)):
                        threads[i].join()
                traci.simulationStep()
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
        if self.config["gui"]:
            sumo_cmd = ["/usr/bin/sumo-gui"]
        else:
            sumo_cmd = ["/usr/bin/sumo"]

        sumo_config = ["-c", "./traffic-sumo/network.sumocfg", '-n', './traffic-sumo/%s' % self.config['net'], '-r', './traffic-sumo/%s' % self.config['route'], 
                      "-a", "./traffic-sumo/%s" % self.config['veh_type'], "-e", str(self.config['end'])]
        sumo_cmd.extend(sumo_config)
        traci.start(sumo_cmd)
        self.traffic_lights = [TrafficLight(config=tl) for tl in self.config['traffic_lights']]
        for tf in self.traffic_lights:
            tf.loadModel()
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
            self.update()
            for i in range(len(self.traffic_lights)):
                self.traffic_lights[i].update(is_train=False)
                self.traffic_lights[i].log_step()
            traci.simulationStep()
        self.close()


    def close(self, ep=-1):
        """
            close simulation
        """
        self.evaluate(ep)
        self.reset()
        traci.close()

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
        self.log[ep]['tf_logs'] = [tf.log for tf in self.traffic_lights]

        print("\n")
        for metric in metrics:
            val = sum(d[metric] for d in veh_logs) / len(veh_logs)
            print("{0:20}:{1}".format(metric, val))
            self.log[ep][metric] = val
        
        if not os.path.exists(self.config['log_folder']):
            os.makedirs(self.config['log_folder'])
        pickle.dump(self.log, open('%s/log.pkl' % self.config['log_folder'], 'wb'))
    
    