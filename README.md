# SSD_Sim

This repo contains three submodules, i.e., MQSim, sample generator, and io_sim.

### 1. MQSim 
MQSim is inherited from https://github.com/CMU-SAFARI/MQSim with the following modifications:
    
    - adding trace features to track intermedia information, including request response time and waiting time.
    - enabling cache capacity by changing a bug in SSD_Sim/MQSim/src/ssd/Data_Cache_Manager_Flash_Advanced.cpp:329. If no free slot is available, find a dirty slot to evict. Otherwise, write data without caching.
    - changing the Back Pressure Space in SSD_Sim/MQSim/src/exec/SSD_Device.cpp:340, by multiplying the default BPS (4096) with a constant value. Tips: recompile if you change the constant value.

### 2. sample generator
We use two versions of sample generators. While v1 uses a complicated filtering mechanism to ensure the mean of size and inter-arrival time follow given distributions, v2 guarantees the workload balance across initiators.

### 3. io simulator API
We provide an api, a warper on top of MQSim, to interact with the network simulator (NS3).


## Tips:
After compiling MQSim, copy the compiled executable file, workload and config files (MQSim, workload.xml, and ssdconfig.xml) to both sample_generator and io_sim for further use.
