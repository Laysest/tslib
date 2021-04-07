import traci
from traffic_light import TrafficLight
import time
import sumolib
from sumolib.miscutils import getFreeSocketPort
import numpy as np
import sys

INTERVAL = 50

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
        self.vehicles = []
        self.controllers = []
        self.config = config

    def update(self):
        pass

    
    def getEdgesOfNode(self, tfID):
        in_edges = []
        out_edges = []
        for edge in self.edges:
            if edge.getToNode() == tfID:
                in_edges.append(edge)
            # elif edge.get

    def run(self, is_train=False):
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

        if is_train:
            for e in range(50):
                print("Episode: %d" % e)
                traci.start(sumo_cmd)
                # create traffic_lights just once
                if e == 0:
                    self.traffic_lights = [  TrafficLight('node1', traci=traci, config=self.config),
                                            TrafficLight('node2', traci=traci, config=self.config), 
                                            TrafficLight('node3', traci=traci, config=self.config), 
                                            TrafficLight('node4', traci=traci, config=self.config) ]                    
                else:
                    for i in range(len(self.traffic_lights)):
                        self.traffic_lights[i].reset()
                count = 0
                while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
                    traci.simulationStep()
                    for i in range(len(self.traffic_lights)):
                        self.traffic_lights[i].update(is_train=is_train)
                        if count % INTERVAL == 0:
                            self.traffic_lights[i].replay()
                    count += 1
                self.close()
                print("-------------------------")
                print("")

        else:
            traci.start(sumo_cmd)
            self.traffic_lights = [  TrafficLight('node1', traci=traci, config=self.config),
                                    TrafficLight('node2', traci=traci, config=self.config), 
                                    TrafficLight('node3', traci=traci, config=self.config), 
                                    TrafficLight('node4', traci=traci, config=self.config)]
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
                traci.simulationStep()
                for i in range(len(self.traffic_lights)):
                    self.traffic_lights[i].update(is_train=is_train)
            self.close()

    def close(self):
        """
            close simulation
        """
        self.evaluate()
        traci.close()

    def evaluate(self):
        """
           Log results of this episode 
        """
        pass