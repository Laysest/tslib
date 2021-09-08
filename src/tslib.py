from src.environment.environment import Environment
from src.glo_vars import GloVars

class TSLib:
    # constructor of TSLib
    def __init__(self, config):
        """
            config:{
                net: file defining traffic network,
                veh_type: file defining vehicle types,
                route: file defining workloads,
                end: maximum time of simulation,
                gui: True/False to enable gui
            }
        """
        self.env = Environment(config)
        GloVars.config = config

    # call to run the environment
    def run(self):
        self.env.run()
    
    def train(self):
        self.env.train()