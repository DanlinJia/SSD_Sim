import os
import random
import argparse
import multiprocessing
import subprocess
import heapq
import sys
from numpy.core.fromnumeric import size

import pandas as pd
import numpy as np
from datetime import datetime
from random import sample, randint
import matplotlib.pyplot as plt
import statistics

from pandas.tseries import offsets
from pathlib import *

sys.path.append('../io_sim')
from io_sim import *


# run under workload_space

def read_dataframe(file_name: str, column_name: str):
    column_df = pd.read_csv(file_name, names=[column_name])
    return column_df

def get_filenames(mypath: str):
    """
    return all file names in mypath
    """
    f = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        f.extend(filenames)
        break
    return f

def copy_traces_to_sample_folder(synthetic_sample_folder):
    """
    copy traces generated under "./traces" to the sample folder.
    """
    if not os.path.exists(synthetic_sample_folder):
        os.system("mkdir -p {}".format(synthetic_sample_folder))
    trace_files = get_filenames(os.path.join(args.workspace, "traces"))
    for trace_file in trace_files:
        shutil.move(src= os.path.join(args.workspace, "traces", trace_file), 
                    dst=os.path.join(synthetic_sample_folder, trace_file))

def execute_BMAP(input_map: str):
    #random_seed = randint(0, 100000)
    random_seed = 987654321
    process = subprocess.Popen("{}/BMAP-Trace {} {}".format(args.workspace, input_map, random_seed), shell=True, stdout=subprocess.PIPE)
    return process

def generate_samples_from_QMAP(map_list: list, synthetic_sample_folder):
    processes = []
    for qmap in map_list:
        processes.append(execute_BMAP(qmap))
    output = [p.wait() for p in processes]
    for i, status in enumerate(output):
        if status!= 0:
            raise Exception("Btrace fails in generating trace for {}".format(map_list[i]))
    copy_traces_to_sample_folder(synthetic_sample_folder)
    return output

def get_sample_df(synthetic_sample_folder):
    """
    read samples as dataframe, where each column is a set of samples for r/w's arrival_time/size.
    """
    tuple_column = [("W", "InterArrival"), ("W", "Size"), ("R", "InterArrival"), ("R", "Size")]
    sample_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(tuple_column))
    for trace in get_filenames(synthetic_sample_folder):
        for io_type, feature in tuple_column:
            if "{}_{}".format(io_type, feature) in trace:
                sample_df[(io_type, feature)] = read_dataframe(os.path.join(synthetic_sample_folder ,trace), trace)[trace]
                break
    return  sample_df

def get_io_type(r_w_ratio):
    n = random.uniform(0, r_w_ratio + 1)
    if n < 1.0:
        return 'W'
    else:
        return 'R'

def load_analyzer(trace_df, calculate_features=False):
    """
    calculate workload of r/w requests in the generated trace.
    """
    def get_size_and_interarrival(df):
        if len(df)==0:
            return 0, 0
        return df.Size.mean(), (df.ArrivalTime.max()-df.ArrivalTime.min())/len(df)
    write_df = trace_df[trace_df.IOType==1]
    read_df = trace_df[trace_df.IOType==0]
    print("read: {}, write: {}".format(len(read_df), len(write_df)))
    write_size, write_arrival = get_size_and_interarrival(write_df)
    read_size, read_arrival = get_size_and_interarrival(read_df)
    write_load = write_size/write_arrival if write_arrival!=0 else 0
    read_load = read_size/read_arrival if read_arrival!=0 else 0
    if calculate_features:
        return write_size, write_arrival, read_size, read_arrival
    return write_load, read_load


def load_analyzer_stdev(trace_df):
    """
    calculate workload of r/w requests in the generated trace.
    """
    def get_size_and_interarrival_std(df):
        if len(df)==0:
            return 0, 0
        inter_arrival = np.diff(df.ArrivalTime)
        return df.Size.std(), inter_arrival.std()
    write_df = trace_df[trace_df.IOType==1]
    read_df = trace_df[trace_df.IOType==0]
    #print("read: {}, write: {}".format(len(read_df), len(write_df)))
    write_size_std, write_arrival_std = get_size_and_interarrival_std(write_df)
    read_size_std, read_arrival_std = get_size_and_interarrival_std(read_df)
    return write_size_std, write_arrival_std, read_size_std, read_arrival_std
    
    

def print_initiator_workload(trace_df):
    for initiatorid in trace_df.InitiatorID.drop_duplicates().values:
        initiator_df = trace_df[trace_df.InitiatorID==initiatorid]
        write_size, write_arrival, read_size, read_arrival = load_analyzer(initiator_df, True)
        write_size_std, write_arrival_std, read_size_std, read_arrival_std = load_analyzer_stdev(initiator_df)
        write_workload = (write_size * 512 * 8/ (1024 * 1024 * 1024)) / (write_arrival / 1e-9); # unit: sector, ns --> Gbps.
        read_workload = (read_size * 512 * 8/ (1024 * 1024 * 1024)) / (read_arrival / 1e-9); # unit: sector, ns --> Gbps.
        
        # Calculate the coefficient of variation (CV) of write_size, write_arrival, read_size, read_arrival.
        write_size_cv = write_size_std / write_size
        read_size_cv = read_size_std / read_size
        write_arrival_cv = write_arrival_std / write_arrival
        read_arrival_cv = read_arrival_std / read_arrival
        
        # Calculate the squared coefficient of variation (SCV) of write_size, write_arrival, read_size, read_arrival.
        write_size_scv = math.pow(write_size_cv, 2)
        read_size_scv = math.pow(read_size_cv, 2)
        write_arrival_scv = math.pow(write_arrival_cv, 2)
        read_arrival_scv = math.pow(read_arrival_cv, 2)
        
        print("Initiator: {}, write_size: {} Bytes, write_arrival: {} ns, write_workload:{} Gbps, write_size_scv: {}, write_arrival_scv: {}, read_size:{} Byte, read_arrival: {} ns, read_workload:{} Gbps, read_size_scv: {}, read_arrival_scv: {}, " \
        .format(initiatorid, int(write_size*512), int(write_arrival), (write_workload), (write_size_scv), (write_arrival_scv), int(read_size*512), int(read_arrival), (read_workload), (read_size_scv), (read_arrival_scv)))


def print_target_workload(trace_df):
    for targetid in trace_df.TargetID.drop_duplicates().values:
        target_df = trace_df[trace_df.TargetID==targetid]
        # load_analyzer designed for sector size and ns inter-arrival
        write_size, write_arrival, read_size, read_arrival = load_analyzer(target_df, True)
        write_size_std, write_arrival_std, read_size_std, read_arrival_std = load_analyzer_stdev(target_df)
        write_workload = (write_size * 512 * 8/ (1024 * 1024 * 1024)) / (write_arrival / 1e-9); # unit: sector, ns --> Gbps.
        read_workload = (read_size * 512 * 8/ (1024 * 1024 * 1024)) / (read_arrival / 1e-9); # unit: sector, ns --> Gbps.
        
        # Calculate the coefficient of variation (CV) of write_size, write_arrival, read_size, read_arrival.
        write_size_cv = write_size_std / write_size
        read_size_cv = read_size_std / read_size
        write_arrival_cv = write_arrival_std / write_arrival
        read_arrival_cv = read_arrival_std / read_arrival
        
        # Calculate the squared coefficient of variation (SCV) of write_size, write_arrival, read_size, read_arrival.
        write_size_scv = math.pow(write_size_cv, 2)
        read_size_scv = math.pow(read_size_cv, 2)
        write_arrival_scv = math.pow(write_arrival_cv, 2)
        read_arrival_scv = math.pow(read_arrival_cv, 2)
        
        print("Target: {}, write_size: {} Bytes, write_arrival: {} ns, write_workload:{} Gbps, write_size_scv: {}, write_arrival_scv: {}, read_size:{} Byte, read_arrival: {} ns, read_workload:{} Gbps, read_size_scv: {}, read_arrival_scv: {}, " \
        .format(targetid, int(write_size*512), int(write_arrival), (write_workload), (write_size_scv), (write_arrival_scv), int(read_size*512), int(read_arrival), (read_workload), (read_size_scv), (read_arrival_scv)))
        

parser = argparse.ArgumentParser(description="generate synthetic traces from QMAPs.")
parser.add_argument("--initiator_num", "-i", type=int, help="the number of initiators.")
parser.add_argument("--target_num", "-t", type=int, help="the number of targets.")
parser.add_argument("--request_num", "-n", type=int, help="the number of requests to generate.")
parser.add_argument("--r_w_ratio", "-r", type=float, default=1.0, help="read write ratio.")
parser.add_argument("--sample_folder", "-sf", type=str, help="the folder where samples are saved.")
parser.add_argument("--trace_name", "-tn", type=str, help="the final generated trace name.")
parser.add_argument("--workspace", "-w", type=str, help="the folder contains QMAPs.", 
                    default=os.path.abspath("./"))
parser.add_argument("--read_arrival_time", "-rt", type=str, 
                    help="the QMAP name of read inter-arrival time.", 
                    default="V0_MAP_R_InterArrival.txt")
parser.add_argument("--write_arrival_time", "-wt", type=str, 
                    help="the QMAP name of write inter-arrival time.",
                    default="V0_MAP_W_InterArrival.txt")
parser.add_argument("--read_size", "-rs", type=str, 
                    help="the QMAP name of read size.",
                    default="V0_MAP_R_Size.txt")
parser.add_argument("--write_size", "-ws", type=str, 
                    help="the QMAP name of write size.",
                    default="V0_MAP_W_Size.txt")

args = parser.parse_args()

# e.g., "Test" is the name of the new created folder.
# Make sure to change the name of the four QMAPs that are used in generating the traces.
# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_InterArrival_1_128_MEAN_10_to_1_scv_test -tn V0_MAP_InterArrival_1_128_MEAN_10_to_1_scv_test -i 10 -t 1 -n 10000

# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_InterArrival_1_64_MEAN_10_to_1 -tn V0_MAP_InterArrival_1_64_MEAN_10_to_1 -i 10 -t 1 -n 10000

# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_InterArrival_R_14us_W_12us_10_to_1_scv_test -tn V0_MAP_InterArrival_R_14us_W_12us_10_to_1_scv_test -i 10 -t 1 -n 10000

# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_InterArrival_R_12us_W_20us_10_to_1_target -tn V0_MAP_InterArrival_R_12us_W_20us_10_to_1_target -i 10 -t 1 -n 10000

# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_R_InterArrival_1_32_MEAN_1_to_1 -tn V0_MAP_R_InterArrival_1_32_MEAN_1_to_1 -i 1 -t 1 -n 10000

# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_R_InterArrival_1_64_MEAN_1_to_1 -tn V0_MAP_R_InterArrival_1_64_MEAN_1_to_1 -i 1 -t 1 -n 10000

# python3 -i ~/ssd-net-sim/scripts/synthetic_generator_v2.py -sf V0_MAP_R_InterArrival_1_128_MEAN_1_to_1 -tn V0_MAP_R_InterArrival_1_128_MEAN_1_to_1 -i 1 -t 1 -n 10000




synthetic_sample_folder = os.path.join(args.workspace, "traces" , args.sample_folder)
maps = [args.read_arrival_time, args.write_arrival_time, args.read_size, args.write_size]
# generate samples from QMAP and save in ./traces folder
output_status = generate_samples_from_QMAP(maps, synthetic_sample_folder)  # Generate four sample files.
print("Sample generated.")

# The time we get is inter-arrival time.
sample_df = get_sample_df(synthetic_sample_folder)
print("Sample dataframe generated.")

# Columns in the trace file (.trace / .csv)
# offset --> MQSim default five columns as input format ("ArrivalTime", "IOType", "Size", "VolumeID", "Offset").
names = ["RequestID", "ArrivalTime", "IOType", "Size", "VolumeID", "Offset", "InitiatorID", "TargetID"]
trace_df = pd.DataFrame(columns=names)

# Declare and initialize lists for priority queue.
# To balance the read / write initiator/target.
# Priority --> 0 as the first priority.
initiator_r_list = [(0, i) for i in range(args.initiator_num)]
initiator_w_list = [(0, i) for i in range(args.initiator_num)]
target_r_list = [(0, t) for t in range(args.target_num)]
target_w_list = [(0, t) for t in range(args.target_num)]

# i_ts_list = list(range(len(args.initiator_num)))
# ts --> timestamp used to calculate the arrival time (=0 + inter-arrival).
# Update the arrival time on read/write initiators.
r_ts_list = [0 for t in range(args.initiator_num)]
w_ts_list = [0 for t in range(args.initiator_num)]

# Aggregated-size based heapify. Update in each iteration.
heapq.heapify(initiator_r_list)
heapq.heapify(initiator_w_list)
heapq.heapify(target_r_list)
heapq.heapify(target_w_list)


# the initial start timestamp, unit is ns
# ts = 0
# Update the previous blank .trace file.
# e.g. req_idx = [1, 10000]
for req_idx in range(args.request_num):
    io_type = get_io_type(args.r_w_ratio)
    if io_type=="W":
        initiator_list = initiator_w_list
        target_list = target_w_list
        ts_list = w_ts_list
    else:
        initiator_list = initiator_r_list
        target_list = target_r_list
        ts_list = r_ts_list

    # src_size --> initiator size (minimum); dst_size --> target size (minimum).
    src_size, src_idx = initiator_list[0]
    dst_size, dst_idx = target_list[0]
    
    # Generate two random integers to get inter-arrival and size from the sample_df.
    # 0 --> InterArrival (t); 1 --> Size (s).
    t_rnd, s_rnd = np.random.randint(low=0, high=len(sample_df), size=2)
    t = sample_df[(io_type, "InterArrival")].iloc[t_rnd]  # unit is ms
    t = int(t*1e6)  # unit is ns
    # ensure no duplicates in InterArrival Time
    while (ts_list[src_idx] + t) in trace_df.ArrivalTime.values:
        t_rnd = np.random.randint(low=0, high=len(sample_df))
        t = sample_df[(io_type, "InterArrival")].iloc[t_rnd]  # unit is ms
        t = int(t*1e6)  # unit is ns
    # Arrival time for initiator src_idx
    ts_list[src_idx] += t
    s = int(sample_df[(io_type, "Size")].iloc[s_rnd])  # unit is sector
    s = 1 if s==0 else s

    # Update current initiator's and target's size.
    new_src_size = src_size + s
    new_dst_size = dst_size + s

    # MQSim default value to calculate SSD store-address.
    offset = random.randint(0, 2.5*1024*1024) * 4 * 1024    # unit is byte, at granularity of 4KB pages
    io_type = 1 if io_type=="W" else 0
    # Update the trace file by adding all the values.
    trace_df.loc[len(trace_df), :] = [req_idx, ts_list[src_idx], io_type, s, src_idx, offset, src_idx, dst_idx]
    ## names = ["RequestID", "ArrivalTime", "IOType", "Size", "VolumeID", "Offset", "InitiatorID", "TargetID"].
    ## "VolumeID" = "InitiatorID".
    
    heapq.heapreplace(initiator_list, (new_src_size, src_idx))
    heapq.heapreplace(target_list, (new_dst_size, dst_idx))

# ensure no duplicates 
assert(len(trace_df.ArrivalTime.drop_duplicates())==args.request_num)
# MQSim requires ArrivalTime is linearly increased. So, first sort the ArrivalTime list. 
trace_df = trace_df.sort_values(by=["ArrivalTime"], ignore_index=True)
# Update RequestID in ascending order [1, 10000]. The RequestID is different from the previous indeices.
trace_df.RequestID = trace_df.index
print("Trace generated")

print_initiator_workload(trace_df)
print("#########################################")
print_target_workload(trace_df)

print("#########################################")

ssd_in_trace="io_sim_trace/net-ssd-{}/ssd.trace".format(args.sample_folder)
net_out_trace = "io_sim_trace/net-ssd-{}/net_out_trace".format(args.sample_folder)
output_folder = "io_sim_trace/net-ssd-{}".format(args.sample_folder)
os.system("mkdir {}".format(output_folder))
output_name = "{}.csv".format(args.trace_name)
workload_path = os.path.abspath("workload.xml")
ssdconfig_path = os.path.abspath("ssdconfig.xml")

# Construct the ssd simulator to run iteration 0. 
# .csv VS. .trace --> DelayTime and FinishTime for SSD.
sm = ssd_simulator(ssd_in_trace, net_out_trace, output_folder, output_name, workload_path, ssdconfig_path)

trace_path = os.path.join(output_folder, "{}.trace".format(args.trace_name))
trace_df.to_csv(path_or_buf=trace_path, sep=",", header=True, index=False)

output_path = os.path.join(output_folder, "{}.csv".format(args.trace_name))
output_df = sm.initialize_SSD_trace(trace_path)
output_df.to_csv(path_or_buf=output_path, sep=",", header=True, index=False)