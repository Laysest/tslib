"""
    This file is an simple example to apply available TSCs to a traffic network.
"""
from tslib import TSLib

# config = {
#     'net': 'isolated.net.xml',
#     'veh_type': 'type.xml',
#     'route': 'isolated.1.route.xml',
#     'end': 2000,
#     'traffic_lights':[
#         {'node_id': 'node1', 'method': 'CDRL', 'yellow_duration': 3, 'cycle_control': 5}
#     ],
#     'gui': False
# }

config = {
    'net': '4x1-two-way.net.xml',
    'veh_type': 'type.xml',
    'route': '4x1-two-way.light.route.xml',
    'end': 3600,
    'traffic_lights': [
        {'node_id': 'node1', 'method': 'CDRL', 'yellow_duration': 3, 'cycle_control': 5},
        {'node_id': 'node2', 'method': 'CDRL', 'yellow_duration': 3, 'cycle_control': 5},
        {'node_id': 'node3', 'method': 'CDRL', 'yellow_duration': 3, 'cycle_control': 5},
        {'node_id': 'node4', 'method': 'CDRL', 'yellow_duration': 3, 'cycle_control': 5},
    ],
    'gui': False
}

sim = TSLib(config)
sim.run(is_train=True)
