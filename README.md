# tslib
TSLib: A library for Traffic Signal Control using deep reinforcement learning and Benchmarking 

This library offers testing Traffic Signal Controls (TSC) including both traditional and advanced methods.

## How to install:
...

## How to use:
Currently, the library is able to use in a large-scale traffic network of a widely used traffic simulator named **SUMO** <sup>[1]</sup>. To use this lirary, you need to provide several required files to obtain a SUMO simulation including (also you can file our example in the source):

- network file (.net.xml)
- vehicle types' defination file (.type.xml)
- route file (.route.xml)

There are some parameters for an signalized intersection:

- node_id: (string) id of a node (e.g. intersection) which you want to apply TSC
- traffic_light_id: (string) the traffic light id of the intersection
- method: TSC methods including "SOTL<sup>[2]</sup>", "MaxPressure<sup>[3]</sup>", "DPGTSC<sup>[4]</sup>", "IntelliLight<sup>[5]</sup>", "3DQNTSC<sup>[6]</sup>"
- yellow_duration: time length of the yellow phases
- cycle_control: the period between two actions making

An example of using tslib:
```
import tslib

traffic_lights = [{'node_id': 'node_1', 'traffic_light_id': 'node_1', 'method': 'Webster', 'yellow_duration': 3, 'cycle_control': 5}]

simulation = tslib.simulation(net= 'netfile.net.xml', veh_type= 'vehtype.net.xml', route= 'routefile.route.xml', step_end= 3600)

simulation.start()
```

To use our library, please cite our paper:
```
{
    @article: 
}
```

*******
### References:
[1] "Microscopic Traffic Simulation using SUMO"; Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann, Yun-Pang Flötteröd, Robert Hilbrich, Leonhard Lücken, Johannes Rummel, Peter Wagner, and Evamarie Wießner. IEEE Intelligent Transportation Systems Conference (ITSC), 2018

[2] C. Gershenson, “Self-organizing traffic lights,”Complex Systems, vol. 16, no. 1, 2004.

[3] P. Varaiya, “The max-pressure controller for arbitrary networks of signalized intersec-tions,” 2013.

[4] S. Mousavi, M. Schukat, P. Corcoran, and E. Howley, “Traffic Light Control Us-ing Deep Policy-Gradient and Value-Function Based Reinforcement Learning,”IETIntelligent Transport Systems, vol. 11, 2017.

[5] H. Wei, G. Zheng, H. Yao, and Z. Li, “Intellilight: A reinforcement learning approachfor intelligent traffic light control,” inProceedings of the 24th ACM SIGKDD Inter-national Conference on Knowledge Discovery & Data Mining, pp. 2496–2505, ACM,2018.

[6] X. Liang, X. Du, G. Wang, and Z. Han, “A deep reinforcement learning network fortraffic light cycle control,”IEEE Transactions on Vehicular Technology, vol. 68, no. 2,pp. 1243–1253, 2019
