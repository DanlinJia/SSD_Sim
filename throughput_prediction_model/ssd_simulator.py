#!/usr/bin/python

import pandas as pd
import os
import numpy as np
from numpy import random
import math
import multiprocessing

from xml.etree import ElementTree as et

INITIATOR_NUMBER = 100
TARGET_NUMBER = 100


def get_synthetic_time(df, scale):
    """
    Generate ArrivalTimele at nanosecond if scale == 1e9
                            microsecond if scale == 1e6
                            milisecond if scale == 1e3
    """
    size = df.groupby(['ArrivalTime']).size()
    ret = np.array([], dtype=int)
    for i in range(len(size)):
        l = i * scale
        r = (i+1) * scale
        synthetic = random.uniform(l, r, size[i])
        synthetic = np.array(sorted(synthetic), dtype=int)
        ret = np.concatenate((ret, synthetic))
    return ret

def request_sampling(df, ratio=0.5):
    df = df.sample(frac=ratio, replace=False, random_state=1).sort_index()
    return df

def read_trace_df(path):
    df = pd.read_csv(path, names=["ArrivalTime", "VolumeID", "Offset", "Size", "IOType"], sep=" ")
    return df

def generate_MQSim_trace(df, ommit_write, trace_folder, host_id = 0):
    df = df[df["TargetID"]==host_id]
    #reindex column to desired order required by MQSim
    #1.Request_Arrival_Time 2.Device_Number 3.Starting_Logical_Sector_Address 4.Request_Size_In_Sectors 5.Type_of_Requests[0 for write, 1 for read]
    names = ["ArrivalTime", "VolumeID", "Offset", "Size", "IOType"]
    df = df.reindex(columns=names)
    df["IOType"] = df["IOType"].apply(lambda x: x^1)
    min_volume = df["VolumeID"].min()
    df["0based_VolumeID"] = df["VolumeID"].apply(lambda x: x - min_volume)

    if ommit_write:
        df = df[df["IOType"]==1]

    df.index = np.array(range(len(df)))
    output_prefix = "{}_{}_".format("host", host_id)
    trace_prefix = {0: output_prefix + "read", 1: output_prefix + "write", 2: output_prefix + "all"}
    if ommit_write:
        trace_name = trace_prefix[0]
    else:
        trace_name = trace_prefix[2]
    trace_path = "{}/{}".format(trace_folder, trace_name)
    names = ["ArrivalTime", "0based_VolumeID", "Offset", "Size", "IOType"]
    df[names].to_csv(path_or_buf=trace_path, sep=" ", header=False, index=False)
    return df, trace_path

def generate_target_id_fixed_target(df):
    #generate 0-based host id
    volume2index = {}
    volumes = df["VolumeID"].drop_duplicates().values
    volumes.sort()
    index = np.array(range(len(volumes)))
    for i in index:
        volume2index[volumes[i]] = i

    df["VolumeID"] =  df["VolumeID"].apply(lambda x: volume2index[x])
    df["TargetID"] = (df["VolumeID"]/(len(index)/TARGET_NUMBER)).astype(int)
    return df

def generate_target_id_dynamic_target(df, volumes_per_target):
    #generate 0-based host id dynamicly
    df["TargetID"] = (df["VolumeID"]/(volumes_per_target)).astype(int)
    df["TargetID"] = df["TargetID"] - df["TargetID"].min()
    return df

def generate_target_id_load_balance(df, target_number):
    volume_df = df[["VolumeID", "Size"]]
    size_df = volume_df.groupby(["VolumeID"]).sum().sort_values(by=["Size"]).reset_index()
    volume2target = {}
    for i in range(len(size_df)):
        volume2target[size_df.VolumeID.iloc[i]] = i%target_number
    df["TargetID"] = df["VolumeID"].apply(lambda x: volume2target[x])
    return df

def remove_large_request(df, threshold=524288):
    volumes = df[df.Size>=threshold]["VolumeID"].drop_duplicates().values
    df = df[~df.VolumeID.isin(volumes)].reset_index()
    return df

def read_Tencent_CBS(file_name):
    #give each column name
    #IOType is "Read(0)", "Write(1)".
    names = ["ArrivalTime", "Offset", "Size", "IOType", "VolumeID"]
    df = pd.read_csv(file_name, names=names)
    return df

def read_VDI_trace(file_name):
    names = ['ArrivalTime', 'ResponseTime', 'IOType', 'LUN', 'Offset', 'Size']
    df = pd.read_csv(file_name, names=names, skiprows=[0])
    df["ArrivalTime"] = df["ArrivalTime"].apply(lambda x: x - df["ArrivalTime"].iloc[0])
    df["Time(ms)"]=(df.ArrivalTime*1000).astype(int)
    df["Time(s)"]=(df.ArrivalTime).astype(int)
    return df

def read_K5cloud_trace(file_name):
    #format: ID, Timestamp, Type, Offset, Length
    names = ["VolumeID", "ArrivalTime", "IOType", "Offset", "Size"]
    df = pd.read_csv(file_name, names=names)
    df["Time(ms)"]=(df.ArrivalTime*1000).astype(int)
    df["Time(s)"]=(df.ArrivalTime).astype(int)
    return df

def get_df_from_input(file_name="data/xan"):
    df = read_Tencent_CBS(file_name)
    df["ArrivalTime"] = df["ArrivalTime"].apply(lambda x: x - df["ArrivalTime"].iloc[0])
    df = df[df["ArrivalTime"]>=0]
    df["ArrivalTime"] = get_synthetic_time(df, 1e9)
    df = generate_target_id_dynamic_target(df, 250)
    #generate_target_id_load_balance(df, TARGET_NUMBER)
    #df = request_sampling(df, ratio=0.2)
    #df = remove_large_request(df, threshold=524288)
    df.index = np.array(range(len(df)))
    return df

def generate_user_id(df, max_user_num):
    #users = np.random.randint(0, max_user_num+1, size=len(df))
    volumes = df["VolumeID"].drop_duplicates().values
    volume2user = {}
    for i in volumes:
        volume2user[i] = random.randint(0, max_user_num)
    df["InitiatorID"] = df["VolumeID"].apply(lambda x: volume2user[x]) 
    return df

def generate_output(trace_df, response_df, output_file):
    output_df = pd.concat([trace_df, response_df], axis=1, sort=False)
    output_df["FinishTime"] = output_df["ArrivalTime"] + output_df["DelayTime"]
    output_df.loc[:, "Size"] = output_df["Size"].apply(lambda x: x*512)
    return output_df

# def get_response_df(host_id, response_file = "workload-trace.IO_Flow.No_0.log"):
#     df_res = pd.read_csv(response_file, sep="\t", header=0)
#     df_res["DelayTime"] = df_res["EndToEndDelay(us)"]/1e6
#     df_res["TargetID"] = np.array([host_id]*len(df_res))
#     return df_res[["DelayTime", "TargetID"]]

def get_response_df(host_id, response_file = "response"):
    df_res = pd.read_csv(response_file, names=["DelayTime"])
    #df_res["TargetID"] = np.array([host_id]*len(df_res))
    os.system("rm {}".format(response_file))
    return df_res

def generate_save_output(trace_df, response_df, output_file):
    output_df = pd.concat([trace_df, response_df], axis=1, sort=False)
    output_df["FinishTime"] = output_df["ArrivalTime"] + output_df["DelayTime"]
    output_df["RequestID"] = output_df.index
    output_df.to_csv(path_or_buf=output_file, sep=",", header=True, index=False)
    return output_df

def run_MQSim(MQSim_input_trace, MQSim_output_folder):
    workload = "workload-trace.xml"
    tree = et.parse(workload)
    tree.find('IO_Scenario/IO_Flow_Parameter_Set_Trace_Based/File_Path').text = MQSim_input_trace
    tree.write(workload)
    os.system("./MQSim -i ssdconfig.test.xml -w {}".format(workload))
    response_file = "{}/res_{}".format(MQSim_output_folder, MQSim_input_trace.split("/")[-1])
    trace_statistic = "{}/statistic_{}".format(MQSim_output_folder, MQSim_input_trace.split("/")[-1])
    #os.system("mv testwrite.csv {}".format(response_file))
    os.system('cp workload-trace_scenario_1.xml {}'.format(trace_statistic))
    return response_file

def simulate_single_host(overall_df, output_folder ,trace_folder ,ommit_write, host_id):
    #output_prefix = "input_trace/host"
    trace_df, trace_name = generate_MQSim_trace(overall_df, ommit_write, trace_folder, host_id)
    response_file = run_MQSim(trace_name, output_folder)
    response_df = get_response_df(host_id, response_file)
    output_df = generate_save_output(trace_df, response_df, "{}/host_{}.csv".format(output_folder, host_id))
    return output_df

def merge_output(input_df, input_folder, output_file):
    df = pd.DataFrame()
    for i in input_df["TargetID"].drop_duplicates().values:
        tmp_df = pd.read_csv("{}/host_{}.csv".format(input_folder, i))
        df =  pd.concat([df, tmp_df], axis=0, sort=True)
    df = generate_user_id(df, INITIATOR_NUMBER)
    names = ["ArrivalTime", "InitiatorID", "VolumeID", "TargetID" ,"Offset", "Size", "IOType", "DelayTime" ,"FinishTime"]
    df = df.reindex(columns=names)
    df["Size"] =  df["Size"]*512
    df.to_csv(path_or_buf=input_folder+output_file, sep=",", header=True, index=False)
    return df

def simulate_hosts(ommit_write, ASCII_file, trace_folder, output_folder, output_name):
    os.system("mkdir "+trace_folder)
    os.system("mkdir "+output_folder)
    input_df = get_df_from_input(ASCII_file)

    for i in input_df["TargetID"].drop_duplicates().values:
        print("start processing host {}".format(i))
        output_df = simulate_single_host(input_df, output_folder, trace_folder, ommit_write, i)

    df = merge_output(input_df, output_folder, "/"+output_name)
    return df

ommit_write = False
trace_folder = "xan-input_trace"
output_folder = "xan-output_trace_all"

# file_list = []
# for root, dirs, files in os.walk("data", topdown=False):
#     for name in files:
#         if ("2018" not in name) and "xa" in name:
#             file_list.append(os.path.join(root, name))

# for file in file_list:
#     if len(file.split("/"))==5:
#         print("start processing: " + file)
#         trace_folder = file.split("/")[-1] + "-input_trace"
#         output_folder = file.split("/")[-1] + "-output_trace_all"
#         output_df = simulate_hosts(ommit_write, file, trace_folder, output_folder, file.split("/")[-1] + "_100initiator_90hosts.csv")
#run_MQSim("xan-input_trace/host_18_all", "tmp")
#soutput_df = simulate_hosts(ommit_write, "data/xan", trace_folder, output_folder, "xan" + "_100initiator_90hosts.csv")