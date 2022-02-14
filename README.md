# SSD_Sim

This repo conatins three submodule, i.e., MQSim, sample generator and io_sim.

### 1. MQSim 
MQSim is inherated from https://github.com/CMU-SAFARI/MQSim with following modifications:
    
    - adding trace features to track intermedia information, including request response time, waiting time.
    - enabling cache capacity by changing a bug in SSD_Sim/MQSim/src/ssd/Data_Cache_Manager_Flash_Advanced.cpp:329. If no free slot avilable, find a dirty slot to evict, otherwise, write data without caching.
    - changing the Back Pressure Space in SSD_Sim/MQSim/src/exec/SSD_Device.cpp:340, by multipling the default BPS (4096) with a constant value. Tips: recompile if you change the constant value.

### 2. sample generator
We use two version of sample generators. While v1 uses complicated filtering machenism to ensure the mean of size and inter-arrival time follow given distributions, v2 garautees the workload banlance across initiators.

### 3. io simulator API
We provide an api, a warper on top of MQSim, to interact with the network simulator (NS3).
