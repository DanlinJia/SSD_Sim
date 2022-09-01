# from ssd_simulator import *
from datetime import datetime
import shutil
import os
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xml.etree import ElementTree as et
from pathlib import *

class ssd_simulator:
    def __init__(self, ssd_in_trace, net_out_trace, output_folder, output_name, workload_path, ssdconfig_path, ssd_per_target=1):
        #ssd_simulator = ssd_simulator("/home/labuser/ssd-net-sim/Congest/ssd_work_space/ssd_tmp/tmp.trace","/home/labuser/ssd-net-sim/Congest/ssd_work_space/ssd_tmp","net-ssd.csv")
        self.distances = {"DelayTime":[], "FinishTime":[]}
        self.ssd_in_trace = ssd_in_trace
        self.net_out_trace = net_out_trace
        self.output_folder = output_folder
        self.output_name = output_name
        self.output_path = "{}/{}".format(output_folder, output_name)
        self.ssd_per_target = ssd_per_target
        self.workload_path = workload_path
        self.ssdconfig_path = ssdconfig_path

    def set_output_folder(self, output_folder):
        self.output_folder = output_folder
        self.output_path = os.path.join(output_folder, self.output_name)
    
    def set_net_out_trace(self, net_out_trace):
        self.net_out_trace = net_out_trace

    def read_trace_df(self, path):
        df = pd.read_csv(path, names=["ArrivalTime", "VolumeID", "Offset", "Size", "IOType"], sep=" ")
        return df

    def calculate_distance(self, old_array, new_array):
        dist = np.linalg.norm(new_array-old_array)
        dist = np.linalg.norm((new_array - old_array), ord=1)
        return dist

    def generate_output(self, trace_df, response_df, output_file):
        #ToDo: check correctness
        output_df = pd.concat([trace_df, response_df], axis=1, sort=False)
        output_df["FinishTime"] = output_df["ArrivalTime"] + output_df["DelayTime"]
        output_df.loc[:, "Size"] = output_df["Size"].apply(lambda x: x*512)
        return output_df

    def run_MQSim(self, MQSim_input_trace, MQSim_output_folder, targetid, workload_path, ssdconfig_path):
        # modify input of the workload trace
        tree = et.parse(workload_path)
        tree.find('IO_Scenario/IO_Flow_Parameter_Set_Trace_Based/File_Path').text = MQSim_input_trace
        tree.write(workload_path)
        # execute MQSim
        cmd = ["./MQSim", "-i", ssdconfig_path,  "-w", workload_path]
        # os.system("./MQSim -i ssdconfig.test.xml -w {}".format(workload))
        temp = subprocess.Popen(cmd, stdout = subprocess.PIPE)
        output = str(temp.communicate())
        # copy statistics to the output_folder
        trace_statistic = os.path.join("{}".format(MQSim_output_folder),"statistic_target{}_{}".format(targetid, os.path.basename(MQSim_input_trace)))
        shutil.copyfile("{}_scenario_1.xml".format(PurePath(workload_path).stem), "{}".format( trace_statistic))
        with open("{}".format(trace_statistic), "w") as file:
            # TODO: extract info from the statistic file
            pass

    # def run_SSD_sim(self, input_path, output_folder, output_file, trace_df=pd.DataFrame([]), old_df=pd.DataFrame([])):
    #     run_MQSim(input_path, output_folder)
    #     response_df = get_response_df(0)
    #     if len(trace_df)==0:
    #         trace_df = self.read_trace_df(input_path)
    #     output_df = self.generate_output(trace_df, response_df, output_file)
    #     distance = None
    #     if len(old_df)>0:
    #         distance = {}
    #         for column in ["DelayTime", "FinishTime"]:
    #             x1 = output_df.sort_values(by=["RequestID"])[output_df.IOType==1][column]
    #             x2 = old_df.sort_values(by=["RequestID"])[output_df.IOType==1][column]
    #             distance[column] = self.calculate_distance(x1, x2)
    #     return output_df, distance

    def get_response_df(self, response_file = "response"):
        df_res = pd.read_csv(response_file, delimiter=" ", names=["ArrivalTime", "DelayTime"])
        df_res = df_res.sort_values(by=["ArrivalTime"])
        df_res = df_res.reset_index(drop=True)
        os.remove("{}".format(response_file))
        return df_res

    def simulate_target(self, trace_df):
        response_df_list = []
        for targetid in trace_df.TargetID.drop_duplicates().values:
            print("start ssd simulation for target {}".format(targetid))
            target_df = trace_df.loc[trace_df["TargetID"]==targetid, :]
            target_df = target_df[["ArrivalTime", "VolumeID", "Offset", "Size", "IOType"]].copy()
            for ssd_idx in range(self.ssd_per_target):
                ssd_df = target_df[target_df.index % self.ssd_per_target == ssd_idx]  
                ssd_df.to_csv(path_or_buf=self.ssd_in_trace, sep=" ", header=False, index=False)
                if os.path.exists("response"):
                    raise Exception('response file should not exist!')
                # run MQSim
                self.run_MQSim(self.ssd_in_trace, self.output_folder, targetid, self.workload_path, self.ssdconfig_path)
                # get the response_df with two columns: [ArrivalTime, DelayTime], sorted by ArrivalTime
                response_df_list.append(self.get_response_df(response_file = "response"))
        return pd.concat(response_df_list).sort_values(by=["ArrivalTime"])

    def initialize_SSD_trace(self, snis_in_trace):
        trace_df = pd.read_csv(snis_in_trace, header=0)
        trace_df.loc[:, "IOType"] = trace_df.IOType.apply(lambda x: x^1)
        response_df = self.simulate_target(trace_df)
        assert(int((response_df.ArrivalTime.values - trace_df.ArrivalTime.values).mean())==0)
        new_df = trace_df.merge(response_df, left_on='ArrivalTime', right_on='ArrivalTime')
        new_df["FinishTime"] = new_df["ArrivalTime"] + new_df["DelayTime"]
        new_df["Size"] = new_df.Size.apply(lambda x: x*512)
        names = ["RequestID", "ArrivalTime", "DelayTime", "FinishTime", "InitiatorID", "TargetID", "IOType", "Size", "VolumeID", "Offset"]
        new_df = new_df[names].sort_values(by=["ArrivalTime"])
        new_df = new_df.reset_index(drop=True)
        new_df["RequestID"] = new_df.index
        new_df[names].to_csv(path_or_buf=self.output_path, sep=",", header=True, index=False)
        return new_df[names]

    def ssd_simulation_iter(self, arrival_time):
        if os.path.exists("response"):
            raise Exception('response file should not exist!')
        # ssd_in_trace is the temp input file of MQSim
        intermedia_path = self.ssd_in_trace
        # read the output of network simulator
        net_df = pd.read_csv(self.net_out_trace, header=0)
        trace_df = net_df.loc[:, ["RequestID","ArrivalTime", "VolumeID", "Offset", "Size", "IOType", "TargetID"]]
        trace_df = trace_df.sort_values(by=["ArrivalTime"])
        trace_df.loc[:, "Size"] = trace_df.Size.apply(lambda x: int(x/512))
        # change IOType to the opposite for MQSim
        trace_df.loc[:, "IOType"] = trace_df.IOType.apply(lambda x: x^1)
        # ArrivalTime is integer with ns unit
        trace_df.loc[:, "ArrivalTime"] = trace_df["ArrivalTime"].astype(np.int64)
        # response_df_list = []
        # for targetid in trace_df.TargetID.drop_duplicates().values:
        #     print("start ssd simulation for target {}".format(targetid))
        #     target_df = trace_df[trace_df["TargetID"]==targetid]
        #     target_df.drop(["RequestID", "TargetID"], axis=1).to_csv(path_or_buf=intermedia_path, sep=" ", header=False, index=False)
        #     # run MQSim
        #     self.run_MQSim(intermedia_path, self.output_folder, targetid, self.workload_path, self.ssdconfig_path)
        #     # get the response_df with two columns: [ArrivalTime, DelayTime], sorted by ArrivalTime
        #     response_df_list.append(self.get_response_df(response_file = "response"))
        
        response_df = self.simulate_target(trace_df)
        # return response_df, trace_df
        assert(int((response_df.ArrivalTime.values - trace_df.ArrivalTime.values).mean())==0)
        ssd_df = net_df.loc[:, ["RequestID", "ArrivalTime", "DelayTime", "FinishTime", "InitiatorID", "TargetID", "IOType", "Size", "VolumeID", "Offset"]]
        ssd_df = ssd_df.sort_values(by=["ArrivalTime"])
        ssd_df.loc[:, "DelayTime"] = response_df["DelayTime"].values
        ssd_df.loc[:, "FinishTime"] = ssd_df["DelayTime"] + ssd_df["ArrivalTime"]
        ssd_df.loc[:, "ArrivalTime"] = arrival_time
        ssd_df.to_csv(path_or_buf=self.output_path, sep=",", header=True, index=False)
        # print("Throuput(GB/s): {}, Distance(ns): {}, Ave_latency(ns): {}".format((ssd_df.Size.sum()/(ssd_df["FinishTime"].iloc[-1] - ssd_df["ArrivalTime"].iloc[0])*1e9/1024/1024/1024), distance["FinishTime"]/len(ssd_df), ssd_df.DelayTime.mean()))
        names = ["RequestID", "ArrivalTime", "DelayTime", "FinishTime", "InitiatorID", "TargetID", "IOType", "Size", "VolumeID", "Offset"]
        return ssd_df[names]

if __name__=="__main__":
    #workload = "/home/labuser/ssd-net-sim/traces/net-ssd-Test/test.csv"
    #workload =Congest/ssd_work_space/Fujitsu-1W_V0_based_25us_10_to_1_net-ssd.csv
    filename = "test"
    #filename = "Fujitsu-1W_V0_based_25us_10_to_1_net-ssd"
    workload = "/home/labuser/Downloads/disaggregate-storage-simulator/Congest/ssd_work_space/Fujitsu-1W_V0_based_25us_10_to_1_net-ssd.csv"
    ori = pd.read_csv(workload)
    init_arrival = ori.ArrivalTime
    ssd_output = "net-ssd.csv"
    ssd_sim = ssd_simulator("./tmp",
                            workload, "./test",
                            ssd_output, "workload.xml", "ssdconfig.test.xml", 2)
    df = ssd_sim.ssd_simulation_iter(init_arrival)
    
