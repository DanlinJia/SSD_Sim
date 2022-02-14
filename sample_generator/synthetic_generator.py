import os
import math

import multiprocessing
import subprocess

import pandas as pd
import numpy as np
from datetime import datetime
from random import sample, randint
import matplotlib.pyplot as plt
import statistics

from io_sim import *


# run under workload_space

def read_dataframe(file_name: str, column_name: str):
    column = pd.read_csv(file_name, names=[column_name])
    return column

def get_filenames(mypath: str):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        f.extend(filenames)
        break
    return f

def generate_samples(input_map: str, random_seed=2564):
    random_seed = randint(0, 100000)
    process = subprocess.Popen("{}/BMAP-Trace {} {}".format(workload_space, input_map, random_seed), shell=True, stdout=subprocess.PIPE)
    return process

def generate_samples_from_VMAP(volume_str: str, map_list: list):
    v_maps = filter(lambda m: volume_str in m, maps)
    v_maps = list(v_maps)
    processes = []
    for f in v_maps:
        processes.append(generate_samples(f))
    output = [p.wait() for p in processes]

def generate_samples_for_initiator(VMAP: str, initiatorid: int, sythetic_trace_path: str, targetid: int):
    generate_samples_from_VMAP(VMAP, maps)
    sample_path = os.path.join(workload_space, "traces")
    samples = get_filenames(sample_path)
    trace_path= os.path.join(workload_space, sythetic_trace_path)
    if not os.path.exists(trace_path):
        os.system("mkdir {}".format(trace_path))
    for trace_sample in samples:
        os.system("cp {}/{} {}/{}_{}_{}".format(sample_path, trace_sample, trace_path, targetid ,initiatorid, trace_sample))

def generate_samples_from_same_vmap(sample_loops: int, VMAP: str ,sythetic_trace_path: str, targetid=0):
    # n vloumes to 1 target
    for initiatorid in range(sample_loops):
        generate_samples_for_initiator(VMAP, initiatorid, sythetic_trace_path, targetid)

def merge_traces(filenames: list, iotype: str):
    try:
        for filename in filenames:
            if filename.split("_")[-2] == iotype:
                if 'InterArrival' in filename:
                    arrivaltime = read_dataframe(os.path.join(sythetic_trace_path, filename), "ArrivalTime").cumsum()
                    # arrivaltime = arrivaltime * 10
                elif "Size" in filename:
                    size = read_dataframe(os.path.join(sythetic_trace_path, filename), "Size")
        data = [[1 if iotype=="W" else 0]]*len(size)
        iotype_df = pd.DataFrame(data, columns=["IOType"])
        columns = pd.concat([arrivaltime, size, iotype_df], axis=1)
        return columns
    except Exception as e:
        print(e)
        print("filenames: {}".format(filenames))

def generate_synthetic_trace(volume_id: int, targetid: int, filenames: list):
    read_df = merge_traces(filenames, "R")
    write_df = merge_traces(filenames, "W")
    df = pd.concat([read_df, write_df])
    df = df.sort_values(by=["ArrivalTime"])
    df["Offset"] = np.random.uniform(low=0, high=10*1024*1024*1024, size=len(df))
    df["VolumeID"] = np.array([volume_id]*len(df))
    df["TargetID"] = np.array([targetid]*len(df))
    df = df.reset_index()
    df["RequestID"] = df.index
    df = df[["RequestID", "ArrivalTime", "IOType", "Size", "Offset", "VolumeID", "TargetID"]]
    return df

def get_trace_df_for_volume_target(volumeid=0, targetid=0):
    samples = get_filenames(sythetic_trace_path)
    target_samples = list(filter(lambda trace_sample: trace_sample.split("_")[0]==str(targetid), samples))
    volume_samples = list(filter(lambda trace_sample: trace_sample.split("_")[1]==str(volumeid), target_samples))
    # 1. RequestID; 2. Arrival Time; 3. IO Type; 4. Size; 5. Offset; 6. Volume ID 7. TargetID
    df = generate_synthetic_trace(volumeid, targetid, volume_samples)
    return df

def concatenate_volume_df(target, sample_loops: int):
    dfs = []
    #targetid2volumeid = {target_id: cast_ratio}
    for volumeid in range(sample_loops):
        df = get_trace_df_for_volume_target(volumeid=volumeid, targetid=target)
        dfs.append(df)
    trace_df = pd.concat(dfs)
    trace_df = trace_df.sort_values(by=["ArrivalTime"])
    trace_df.index = range(len(trace_df))
    trace_df["RequestID"] = trace_df.index
    return trace_df

def plot_actual_cast_ratio(trace_df, output_folder):
    trace_df["Time(us)"] = (trace_df.ArrivalTime * 1000).astype(int)
    df = trace_df.groupby(["Time(us)"]).count()
    df = trace_df.groupby(["Time(us)"]).agg({"VolumeID": "nunique"})
    ser = pd.Series(df.VolumeID)
    ser.hist(cumulative=True, density=1, bins=1000)
    plt.savefig(output_folder+"/count.png")

def post_process(trace_df):
    trace_df = trace_df[~trace_df["ArrivalTime"].isnull()]
    trace_df = trace_df[~trace_df["Size"].isnull()]
    trace_df["InitiatorID"] = trace_df.VolumeID
    trace_df.loc[:, "ArrivalTime"] = trace_df.ArrivalTime.apply(lambda x: x*1e6).astype(int)
    trace_df.loc[:, "Size"] = trace_df.Size.apply(lambda x: math.ceil(x))
    trace_df.loc[:, "Offset"] = trace_df.Offset.astype(int)
    return trace_df

def get_size_and_interarrival(df):
    if len(df)==0:
        return 0, 0
    return df.Size.sum()*512/len(df), (df.ArrivalTime.max()-df.ArrivalTime.min())/len(df)

def load_analyzer(initiator_df):
    # read_workload = 0
    # write_workload = 0
    write_df = initiator_df[initiator_df.IOType==1]
    read_df = initiator_df[initiator_df.IOType==0]
    # if (len(write_df)==0) or (len(read_df)==0):
    #     return write_workload, read_workload
    write_size, write_arrival = get_size_and_interarrival(write_df)
    read_size, read_arrival = get_size_and_interarrival(read_df)
    return write_size, write_arrival, read_size, read_arrival


workload_space = "/home/labuser/Downloads/disaggregate-storage-simulator/scripts/Q3MAP/workload_space"
maps = get_filenames(workload_space)
# sythetic_trace_folder = "1_3_mean_1x_size_1w-T1125-10:1"
#sythetic_trace_folder = "Fujitsu_1_8_mean_4x_size_1w-V0-1:10"
# sythetic_trace_folder = "V0_MAP_R_54us_W_30us_10_to_1_test"
sythetic_trace_folder = "V0_MAP_InterArrival_1_128_MEAN_1_to_1_topology"
# synthetic_trace_name = "V0_MAP_R_54us_W_30us_10_to_1_test_net-ssd"
synthetic_trace_name = "V0_MAP_InterArrival_1_128_MEAN_1_to_1_topology"
sythetic_trace_path = os.path.join(workload_space, sythetic_trace_folder)
VMAP = "V0"
# targetid2volumeid is a directory, mapping target id to a range of initiators
# targetid2volumeid = {0:range(10)}
targetid2volumeid = {0:range(1)}
# 40 to 4: targetid2volumeid = {0:range(10), 1:range(10,20), 2:range(20,30), 3:range(30, 40)}
# create 1 to m 
# for target in range(10):
#     targetid2volumeid[target] = [0]
# volumeid2targetid = {0:range(10)}
sample_loops = 5

ssd_in_trace="/home/labuser/Downloads/disaggregate-storage-simulator/traces/net-ssd-{}/tmp.trace".format(sythetic_trace_folder)
net_out_trace = "~/ssd-net-sim/traces/net-ssd-{}/tmp".format(sythetic_trace_folder)
output_folder = "/home/labuser/Downloads/disaggregate-storage-simulator/traces/net-ssd-{}".format(sythetic_trace_folder)
os.system("mkdir {}".format(output_folder))
output_name = "{}.csv".format(synthetic_trace_name)

sm = ssd_simulator(ssd_in_trace, net_out_trace, output_folder, output_name)
print("Initializing done")
print("target: {}, each target has {} initiators".format(len(targetid2volumeid), [len(targetid2volumeid[i]) for i in targetid2volumeid]))

def generate_trace_for_one_target(target, initiator_range=range(0,10), step=200, sample_loops=50):
    print("start generating trace for target {}".format(target))
    # find all candidates of samples generated by all initiators
    candidates = pd.DataFrame(columns=["requestid", "initiatorid", "write_size", "write_arrival","read_size", "read_arrival"])
    retry_count = 0
    distance_ratio = 0.2
    # make sure we have enough candidates
    while len(candidates)<len(initiator_range):
        print("retry_count: {}, candidates len: {}, distance_ratio: {}".format(retry_count, len(candidates), distance_ratio))
        retry_count += 1
        # generate a sample pool and save it to trace_path
        generate_samples_from_same_vmap(sample_loops, VMAP, sythetic_trace_path, targetid=target)
        target_df = concatenate_volume_df(target, sample_loops)
        target_df = post_process(target_df)
        trace_path = os.path.join(sythetic_trace_path, sythetic_trace_folder+"_{}_synthetic.csv".format(target))
        target_df.to_csv(path_or_buf=trace_path, sep=",", header=True, index=False)
        # loop over sample pool to get all candidates
        for initiator in range(sample_loops):
            initiator_df = target_df[target_df.InitiatorID==initiator]
            initiator_df = initiator_df.reset_index(drop=True)
            for i in range(0, len(initiator_df), step):
                trace_sample = initiator_df.loc[i:i+step-1,:]
                load = load_analyzer(trace_sample)
                candidates.loc[len(candidates), :] = np.array([int(i), int(initiator), *load])
        # ignore candidates if read/write is empty
        candidates = candidates[(candidates.read_arrival!=0) & (candidates.write_arrival!=0)]
        write_size, write_arrival, read_size, read_arrival = candidates[[ "write_size", "write_arrival","read_size", "read_arrival"]].mean().values
        # release filter condition if retry too many times
        if retry_count % 2 == 0:
            distance_ratio += 0.1
        # filter out candidates
        candidates = candidates[ \
                    candidates.write_arrival.between(round(write_arrival*(1-distance_ratio)), round(write_arrival*(1+distance_ratio))) \
                    & candidates.read_arrival.between(round(read_arrival*(1-distance_ratio)), round(read_arrival*(1+distance_ratio))) \
                    & candidates.write_size.between(round(write_size*(1-distance_ratio)), round(write_size*(1+distance_ratio))) \
                    & candidates.read_size.between(round(read_size*(1-distance_ratio)), round(read_size*(1+distance_ratio)))]

    candidates["write_load"] = candidates.write_size/candidates.write_arrival
    candidates["read_load"] = candidates.read_size/candidates.read_arrival
    candidates["distance"] = ( abs(candidates.write_load - write_size/write_arrival) + abs(candidates.read_load - read_size/read_arrival) ) /2
    # rank candidates by distance
    candidates = candidates.sort_values(by=["distance"]).head(len(initiator_range))
    candidates = candidates.reset_index(drop=True)
    candidates_df = []
    # generate candidate_df by mapping cadidates to real_initiator_id
    for i, real_initiator_id in zip(candidates.index.values, initiator_range):
        initiatorid = candidates.loc[i, "initiatorid"]
        initiator_df = target_df[target_df.InitiatorID==initiatorid]
        initiator_df = initiator_df.reset_index(drop=True)
        sample_index = candidates.loc[i, "requestid"]
        sample_df = initiator_df.loc[sample_index: sample_index+step-1, : ]
        # a = sample_df.loc[df.InitiatorID == int(candidates.loc[i, "initiatorid"]), :]
        sample_df.loc[:]["InitiatorID"] = np.array(len(sample_df)*[real_initiator_id])
        sample_df.loc[:]["ArrivalTime"] = sample_df.ArrivalTime.apply(lambda x: x - sample_df.ArrivalTime.iloc[0])
        # print(sample_df.Size.mean(), (sample_df.ArrivalTime.values[-1] - sample_df.ArrivalTime.values[0])/len(sample_df))
        candidates_df.append(sample_df)

    candidate_df = pd.concat(candidates_df)
    candidate_df = candidate_df.sort_values(by=["ArrivalTime"])
    sample_write_size, sample_write_arrival, sample_read_size, sample_read_arrival = load_analyzer(candidate_df)
    if sample_write_size==0 or sample_write_arrival==0 or sample_read_size==0 or sample_read_arrival ==0:
        return    
    trace_sample = candidate_df.reset_index(drop=True)
    # handle requests with 0 arrivaltime
    trace_sample.loc[trace_sample.ArrivalTime==0, "ArrivalTime"] = trace_sample[trace_sample.ArrivalTime==0].index.values
    trace_sample.loc[~trace_sample.index.isin(trace_sample.ArrivalTime.drop_duplicates().index), "ArrivalTime"] += \
        trace_sample[~trace_sample.index.isin(trace_sample.ArrivalTime.drop_duplicates().index)].index.values
    
    trace_sample = trace_sample.sort_values(by=["ArrivalTime"])
    trace_sample = trace_sample.reset_index(drop=True)
    trace_sample.loc[:]["RequestID"] = trace_sample.index
    read_df = trace_sample[trace_sample.IOType==0]
    write_df = trace_sample[trace_sample.IOType==1]
    print("target: {}, R: len={}, size={}, inter-arrival={}, load={}, range={}".format(target, len(read_df), sample_read_size, sample_read_arrival, sample_read_size/sample_read_arrival*8, [candidates.read_load.min()*8, candidates.read_load.max()*8] ))
    print("target: {}, W: len={}, size={}, inter-arrival={}, load={}, range={}".format(target, len(write_df), sample_write_size, sample_write_arrival, sample_write_size/sample_write_arrival*8, [candidates.write_load.min()*8, candidates.write_load.max()*8] ))
    return trace_sample

target_samples = []
for targetid in targetid2volumeid:
    trace_sample = generate_trace_for_one_target(target=targetid, initiator_range=targetid2volumeid[targetid], step=1000, sample_loops=sample_loops)
    target_samples.append(trace_sample)

assert(len(target_samples)==len(targetid2volumeid))
all_sample = pd.concat(target_samples)

# exchange values in TargetID and InitiatorID
all_sample.loc[:, ["InitiatorID", "TargetID"]] = all_sample.loc[:, ["TargetID", "InitiatorID"]].values

# remove duplicated arrival time
while(len(all_sample.ArrivalTime[all_sample.ArrivalTime.duplicated()])!=0):
    all_sample.loc[all_sample.ArrivalTime.duplicated(), "ArrivalTime"] = sample(range(0, all_sample.ArrivalTime.max()), len(all_sample.ArrivalTime[all_sample.ArrivalTime.duplicated()]))
all_sample = all_sample.sort_values(by=["ArrivalTime"])
all_sample.loc[:, "RequestID"] = range(len(all_sample))

trace_sample = all_sample
sample_path = os.path.join(sythetic_trace_path, sythetic_trace_folder+"_synthetic_samples.trace")
trace_sample.to_csv(path_or_buf=sample_path, sep=",", header=True, index=False)
trace_sample.to_csv(path_or_buf=os.path.join(output_folder, "{}.trace".format(synthetic_trace_name)), sep=",", header=True, index=False)
# start ssd simulation and save new_df to a csv file
new_df = sm.initialize_SSD_trace(sample_path)
for initiatorid in new_df.InitiatorID.drop_duplicates().values:
    initiator_df = new_df[new_df.InitiatorID==initiatorid]
    write_size, write_arrival, read_size, read_arrival = load_analyzer(initiator_df)
    print("Initiator: {}, write_size: {}, write_arrival: {}, read_size:{}, read_arrival: {}," \
    .format(initiatorid, int(write_size/512), int(write_arrival), int(read_size/512), int(read_arrival)))

target_workload_df = pd.DataFrame(columns=["targetid", "write_size", "write_arrival", "read_size", "read_arrival"])
for targetid in new_df.TargetID.drop_duplicates().values:
    target_df = new_df[new_df.TargetID==targetid]
    # load_analyzer designed for sector size and ns inter-arrival
    write_size, write_arrival, read_size, read_arrival = load_analyzer(target_df)
    target_workload_df.loc[len(target_workload_df), : ] = [targetid, write_size/512, write_arrival, read_size/512, read_arrival]
    print("Target: {}, write_size: {}, write_arrival: {}, read_size:{}, read_arrival: {}," \
    .format(targetid, int(write_size/512), int(write_arrival), int(read_size/512), int(read_arrival)))
print("average across targets: write_size: {}, write_arrival: {}, read_size:{}, read_arrival: {}," \
    .format(target_workload_df.write_size.mean(), target_workload_df.write_arrival.mean(), \
     target_workload_df.read_size.mean(), target_workload_df.read_arrival.mean()))

# net_input_path = os.path.join(sythetic_trace_path, sythetic_trace_folder+"_synthetic_samples.csv")
# new_df.to_csv(path_or_buf=net_input_path, sep=",", header=True, index=False)