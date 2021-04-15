from glo_vars import GloVars

traci = GloVars.traci

class Vehicle:
    def __init__(self, vehID, now):
        self.id = vehID
        self.type = traci.vehicle.getTypeID(self.id)
        self.start_time = now
        self.finish_time = 0
        self.update()
            
    def update(self):
        self.cur_lane = traci.vehicle.getLaneID(self.id)
        self.cur_speed = traci.vehicle.getSpeed(self.id)

    def isFinished(self):
        return self.finish_time != 0

    def finish(self, now):
        self.finish_time = now