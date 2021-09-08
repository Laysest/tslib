from src.tslib import TSLib

config = {
    'net': 'isolated-intersection/testbed-a-hour/road.net.xml',
    'veh_type': 'type.xml',
    'route': 'isolated-intersection/testbed-a-hour/flow.route.xml',
    'end': 600,
    'gui': False,
    'simulator': 'SUMO',
    'traffic_lights': [
        {'node_id': 'gneJ1',
        'method': 'CDRL',
        'yellow_duration': 3,
        'cycle_control': 5,
        'folder': './model/most/CDRL2'}
    ],
    'log_folder': './log/dev/testbed-a-hour'
}

env = TSLib(config)
env.train()
