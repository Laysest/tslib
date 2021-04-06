from tslib import TSLib

# config = {
#     'net': '4x1-two-way.net.xml',
#     'veh_type': 'type.xml',
#     'route': '4x1-two-way.light.route.xml',
#     'end': 2000,
#     'gui': True
# }

config = {
    'net': 'isolated.net.xml',
    'veh_type': 'type.xml',
    'route': 'isolated.1.route.xml',
    'end': 2000,
    'gui': True
}

sim = TSLib(config)
sim.run(is_train=False)