from traffic_light import TrafficLight
import time
import sumolib
from sumolib.miscutils import getFreeSocketPort
import numpy as np
import sys
from glo_vars import GloVars
import tensorflow as tf
from Vehicle import Vehicle

traci = GloVars.traci

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

    def reset(self):
        self.vehicles = []

    def update(self):
        now = traci.simulation.getTime()
        vehs_id = [veh.id for veh in self.vehicles]
        for veh_id_ in traci.simulation.getDepartedIDList():
            if veh_id_ not in vehs_id:
                self.vehicles.append(Vehicle(veh, now))

        for veh_id_ in traci.simulation.getArrivedIDList():
            if veh_id_ in vehs_id:
                self.vehicles[vehs_id.index(veh)].finish()

        for i in range(len(self.vehicles)):
            self.vehicles[i].update()
    
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
        pretrain_ = True
        if is_train:

            ### Pre-train: ------------------------
            if pretrain_:
                traci.start(sumo_cmd)
                self.traffic_lights = [ TrafficLight('node1', config=self.config),
                                        TrafficLight('node2', config=self.config), 
                                        TrafficLight('node3', config=self.config), 
                                        TrafficLight('node4', config=self.config) ]
                while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
                    traci.simulationStep()
                    for i in range(len(self.traffic_lights)):
                        self.traffic_lights[i].update(is_train=False, pretrain=True)
                self.close()
            #### ------------------------------------------

            for e in range(50):
                print("Episode: %d" % e)
                traci.start(sumo_cmd)

                # create traffic_lights just once
                if e == 0 and pretrain_ == False:
                    self.traffic_lights = [  TrafficLight('node1', config=self.config),
                                            TrafficLight('node2', config=self.config), 
                                            TrafficLight('node3', config=self.config), 
                                            TrafficLight('node4', config=self.config) ]                    
                else:
                    for i in range(len(self.traffic_lights)):
                        self.traffic_lights[i].reset()
                count = 1
                while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.config['end']:
                    traci.simulationStep()
                    GloVars.step += 1
                    for i in range(len(self.traffic_lights)):
                        self.traffic_lights[i].update(is_train=is_train)
                        if count % GloVars.INTERVAL == 0:
                            self.traffic_lights[i].replay()
                    count += 1
                self.close()
                print("-------------------------")
                print("")

        else:
            traci.start(sumo_cmd)
            self.traffic_lights = [  TrafficLight('node1', config=self.config),
                                    TrafficLight('node2', config=self.config), 
                                    TrafficLight('node3', config=self.config), 
                                    TrafficLight('node4', config=self.config)]
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