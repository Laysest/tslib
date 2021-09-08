from src.tslib import TSLib

config = {
    'simulator': 'CityFlow',
    'config_file': './src/traffic-cityflow/isolated-intersection/hz-bc-tyc_1/config.json',
    'traffic_lights': [
        {
            'node_id': 'intersection_1_1',
            'method': 'CDRL',
            'yellow_duration': 3,
            'cycle_control': 5,
            'folder': './model/most/CDRL2'
        }
    ],
    'end': 3600,
    'log_folder': './model/most/CDRL2'
}

env = TSLib(config)
env.train()