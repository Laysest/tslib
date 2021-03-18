from tslib import TSLib

config = {
    'net': '4x1-one-way.net.xml',
    'veh_type': 'type.xml',
    'route': '4x1-one-way.light.route.xml',
    'end': 3600,
    'gui': True
}


sim = TSLib(config)
sim.run()