import cityflow

config_file = "./traffic-cityflow/isolated-intersection/hz-bc-tyc_1/config.json"
period = 3600

eng = cityflow.Engine(config_file=config_file, thread_num=1)

for _ in range(period):
    eng.next_step()
    vehs = eng.get_vehicles()
    for veh in vehs:
        print(eng.get_vehicle_info(veh))
    # running_count = len()
    total_count = len(eng.get_vehicles(include_waiting=True))
    eng.get_lane_vehicle_count()
    eng.get_lane_waiting_vehicle_count()
    eng.get_lane_vehicles()
    eng.get_vehicle_speed()
    eng.get_vehicle_distance()
    eng.get_current_time()

