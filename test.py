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

# config = {
#     'net': '4x1-two-way.net.xml',
#     'veh_type': 'type.xml',
#     'route': '4x1-two-way.light.route.xml',
#     'end': 3600,
#     'traffic_lights': [
#         {'node_id': 'node1', 'method': 'TLCC', 'yellow_duration': 3, 'cycle_control': 5},
#         {'node_id': 'node2', 'method': 'TLCC', 'yellow_duration': 3, 'cycle_control': 5},
#         {'node_id': 'node3', 'method': 'TLCC', 'yellow_duration': 3, 'cycle_control': 5},
#         {'node_id': 'node4', 'method': 'TLCC', 'yellow_duration': 3, 'cycle_control': 5},
#     ],
#     'gui': True
# }

# config = {
#     'net': 'isolated-intersection/hz-bc-tyc_1/road.net.xml',
#     'veh_type': 'type.xml',
#     'route': 'isolated-intersection/hz-bc-tyc_1/flow.route.xml',
#     'end': 3600,
#     'traffic_lights': [
#         {'node_id': 'intersection_1_1', 'method': 'TLCC', 'yellow_duration': 3, 'cycle_control': 5, 'folder': './model/train/hz-bc-tyc_1/TLCC'},
#     ],
#     'log_folder': './log/train/hz-bc-tyc_1/TLCC',
#     'gui': False,
# }

config = {
    'net': 'arterial_road/atlanta/road.net.xml',
    'veh_type': 'type.xml',
    'route': 'arterial_road/atlanta/flow.route.xml',
    'end': 3600,
    'traffic_lights': [
        {'node_id': '69421277', 'method': 'IntelliLight', 'yellow_duration': 3, 'cycle_control': 5, 'folder': './model/train/atlanta/IntelliLight'},
        {'node_id': '69249210', 'method': 'IntelliLight', 'yellow_duration': 3, 'cycle_control': 5, 'folder': './model/train/atlanta/IntelliLight'},
        {'node_id': '69387071', 'method': 'IntelliLight', 'yellow_duration': 3, 'cycle_control': 5, 'folder': './model/train/atlanta/IntelliLight'},
        {'node_id': '69227168', 'method': 'IntelliLight', 'yellow_duration': 3, 'cycle_control': 5, 'folder': './model/train/atlanta/IntelliLight'},
        {'node_id': '69515842', 'method': 'IntelliLight', 'yellow_duration': 3, 'cycle_control': 5, 'folder': './model/train/atlanta/IntelliLight'},
    ],
    'log_folder': './log/train/atlanta/IntelliLight',
    'gui': False,
}


sim = TSLib(config)
sim.run(is_train=True)
