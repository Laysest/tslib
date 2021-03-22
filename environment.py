import traci
from traffic_light import TrafficLight
import time
import sumolib
from sumolib.miscutils import getFreeSocketPort
import numpy as np


MAX_NUMBER_OF_INTERSECTIONS = 2

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

    def run(self):
        """
            To run a simulation based on the configurate
        """
        if self.config["gui"]:
            sumoCmd = ["/usr/bin/sumo-gui"]
        else:
            sumoCmd = ["/usr/bin/sumo"]

        sumoConfig = ["-c", "./traffic-sumo/network.sumocfg", '-n', './traffic-sumo/%s' % self.config['net'], '-r', './traffic-sumo/%s' % self.config['route'], 
                      "-a", "./traffic-sumo/%s" % self.config['veh_type'], "-e", str(self.config['end'])]
        sumoCmd.extend(sumoConfig)
        traci.start(sumoCmd)
        self.trafficLights = [TrafficLight('node1', traci=traci), TrafficLight('node2', traci=traci), 
                                TrafficLight('node3', traci=traci), TrafficLight('node4', traci=traci)]

        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
            traci.simulationStep()
            for i in range(len(self.trafficLights)):
                self.trafficLights[i].update()
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