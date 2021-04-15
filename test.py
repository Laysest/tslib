"""
    This file is an simple example to apply available TCSs to a traffic network.
"""
from tslib import TSLib

# config = {
#     'net': '4x1-two-way.net.xml',
#     'veh_type': 'type.xml',
#     'route': '4x1-two-way.light.route.xml',
#     'end': 2000,
#     'gui': True
# }

config = {
    'net': '4x1-two-way.net.xml',
    'veh_type': 'type.xml',
    'route': '4x1-two-way.light.route.xml',
    'end': 2000,
    'gui': False
}

sim = TSLib(config)
sim.run(is_train=True)
