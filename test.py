from tslib import TSLib

config = {
    'net': '4x1-two-way.net.xml',
    'veh_type': 'type.xml',
    'route': '4x1-two-way.light.route.xml',
    'end': 3600,
    'gui': True
}


sim = TSLib(config)
sim.run()