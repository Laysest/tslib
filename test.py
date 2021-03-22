from tslib import TSLib

config = {
    'net': 'isolated.net.xml',
    'veh_type': 'type.xml',
    'route': 'isolated.0.route.xml',
    'end': 3600,
    'gui': False
}


sim = TSLib(config)
sim.run(isTrain=True)