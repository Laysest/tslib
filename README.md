# TSLib: A unified Traffic Signal Control framework using deep reinforcement learning and Benchmarking 

*Feel free to reach me at toan.tranviet@ieee.org if you have any issues or need help to run the code*

This framework offers training/testing Traffic Signal Controls (TSC) including both traditional and advanced methods.

## Class diagram:
![Alt text](class_diagram.png?raw=true "TSLIB's Class Diagram")

## Result:
![Alt text](result.png?raw=true "TSLIB for intersections at Monaco")

## How to install:
Using docker:
> docker pull viettoantran98/tslib:0.0.1

Clone the updated source:
> git clone https://github.com/Laysest/tslib

## How to use:
See in some examples.

### SUMO
```
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
# env.run()
```
Where:
* 'net': the road structure file
* 'veh_type': defines vehicle characteristics
* 'route': the workload file
* 'end': the maximum step of simulation
* 'gui': enable GUI or not
* 'simulator': currently SUMO and CityFlow are supported
* 'traffic_lights': an array, each element is used for one intersection
* 'log_folder': for log

> python3 SUMO_one_intersection_Chattanooga.py

### CityFLow:
```
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
# env.run()
```
Where:
* 'config_file': the configuration of your simulation, including information of road structures and workloads


If you use the framework, please cite our paper:
```
@INPROCEEDINGS{9671993,
  author={Tran, Toan V. and Doan, Thanh-Nam and Sartipi, Mina},
  booktitle={2021 IEEE International Conference on Big Data (Big Data)}, 
  title={TSLib: A Unified Traffic Signal Control Framework Using Deep Reinforcement Learning and Benchmarking}, 
  year={2021},
  pages={1739-1747},
  doi={10.1109/BigData52589.2021.9671993}}

```

## Available TSC methods:
<ol>
<li> **FT** Fixed Time

<li> **SOTL** C. Gershenson, “Self-organizing traffic lights,”Complex Systems, vol. 16, no. 1, 2004.

<li> **MaxPressure** P. Varaiya, “The max-pressure controller for arbitrary networks of signalized intersections,” 2013.

<li> **VFB** S. Mousavi, M. Schukat, P. Corcoran, and E. Howley, “Traffic Light Control Using Deep Policy-Gradient and Value-Function Based Reinforcement Learning,”IETIntelligent Transport Systems, vol. 11, 2017.
   
<!-- <li> Krajzewicz, D., Hertkorn, G., Ringel, J., & Wagner, P. (2005). Preparation of digital maps for traffic simulation; Part 1: Approach and algorithms. 3rd International Industrial Simulation Conference 2005, ISC 2005, 285–290.
 -->
    
<li> **IntelliLight** H. Wei, G. Zheng, H. Yao, and Z. Li, “Intellilight: A reinforcement learning approachfor intelligent traffic light control,” inProceedings of the 24th ACM SIGKDD Inter-national Conference on Knowledge Discovery & Data Mining, pp. 2496–2505, ACM,2018.

<li> **TLCC** X. Liang, X. Du, G. Wang, and Z. Han, “A deep reinforcement learning network for traffic light cycle control,”IEEE Transactions on Vehicular Technology, vol. 68, no. 2,pp. 1243–1253, 2019
    
<li> **CDRL** Van der Pol, E., & Oliehoek, F. A. (2016). Coordinated deep reinforcement learners for traffic light control. 30th Conference on Neural Information Processing Systems, Nips, 2018.

<li> **CAREL** Rodrigues, F., & Azevedo, C.L. (2019). Towards Robust Deep Reinforcement Learning for Traffic Signal Control: Demand Surges, Incidents and Sensor Failures. 2019 IEEE Intelligent Transportation Systems Conference (ITSC), 3559-3566.
 
</ol>
