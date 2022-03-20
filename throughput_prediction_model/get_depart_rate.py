from builtins import int, range, sum
from cmath import log
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import pathlib as pl


def get_trace_files(log_folder, inner_key_func=lambda f: True, outer_key_func=lambda f: False):
    trace_files = []
    for root, dirs, files in os.walk(log_folder):
        for f in files:
            if inner_key_func(f) and not outer_key_func(f):
                trace_files.append(os.path.join(root, f))
    trace_files.sort()
    return trace_files

def get_statistic_df(log_folder='/home/labuser/Downloads/MQSim/logs/short_lat_test'):
    # multi-level column can be created from tuples, regardless of what the original column names, e.g., ['a', 'b', 'c', 'd', 'e']
    ret_df = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    ret_df.columns=pd.MultiIndex.from_tuples([("test", ""), ("read", "throughput"), ("read", "ave_lat"), ("read", "last_lat"), ("read", "90th_lat"), ("write", "throughput"), ("write", "ave_lat"), ("write", "last_lat"), ("write", "90th_lat")])
    for root, dirs, files in os.walk(log_folder):
        for f in files:
            if "results.csv" in f:
                path = os.path.join(root, f)
                df = pd.read_csv(path)
                r_df=df[df.IOType==0]
                w_df=df[df.IOType==1]
                last_r_idx = r_df.FinishTime.idxmax()
                last_w_idx = w_df.FinishTime.idxmax()
                # print(f, "read finish rate: {}, write finish rate: {}".format(np.diff(r_df.FinishTime.values).mean()/1e3, np.diff(w_df.FinishTime.values).mean()/1e3))
                #print(f, "read tpt: {} GB/s, write tpt: {} GB/s".format(r_df.Size.sum()/(r_df.FinishTime.iloc[len(r_df)-1]), w_df.Size.sum()/(w_df.FinishTime.iloc[len(w_df)-1]) ))
                ret_df.loc[len(ret_df), :] = [f, r_df.Size.sum()/(r_df.FinishTime.max() - r_df.ArrivalTime.min()), r_df.DelayTime.mean()/1e3, r_df.loc[last_r_idx, "DelayTime" ]/1e3, r_df.DelayTime.quantile(0.9)/1e3 \
                                    ,w_df.Size.sum()/(w_df.FinishTime.max()-w_df.ArrivalTime.min()), w_df.DelayTime.mean()/1e3, w_df.loc[last_w_idx, "DelayTime" ]/1e3,  w_df.DelayTime.quantile(0.9)/1e3] 

    ret_df = ret_df.sort_values(by=["test"], ignore_index=True)
    print(ret_df)
    return ret_df

def plot_runtime_throughput(log_folder, inner_key_func=lambda f: True, outer_key_func=lambda f: False, bucket_size=1e6):
    figure, axs = plt.subplots(2)
    trace_dirt = {}
    trace_files = get_trace_files(log_folder, inner_key_func, outer_key_func)
    # trace_files.sort(key=lambda x: int(pl.PurePath(x).stem.split("_")[1][:-2]))
    for f in trace_files:
        df = pd.read_csv(f)
        df = df.sort_values(by="FinishTime", ignore_index=True)
        df["runtime_throughput"] = (df.Size * 8 / (1024*1024*1024)) / (bucket_size * 1e-9) # Gbps
        df["time_ms"] = (df.FinishTime/bucket_size).astype(int) + 1 # bucket size ns
        r_df=df[df.IOType==0]
        w_df=df[df.IOType==1]
        r_tpt = r_df.groupby(["time_ms"]).sum()
        w_tpt = w_df.groupby(["time_ms"]).sum()
        trace_dirt[f] = [r_tpt.runtime_throughput, w_tpt.runtime_throughput]
        marker = ''
        linestyle = "solid"
        axs[0].plot(r_tpt.runtime_throughput, label=os.path.basename(f).split(".")[0], marker=marker, linestyle=linestyle)
        axs[1].plot(w_tpt.runtime_throughput, label=os.path.basename(f).split(".")[0], marker=marker, linestyle=linestyle)

    axs[0].set_ylabel("throughput (Gbps)")
    axs[0].set_title("read")
    # axs[0].set_yscale("log")
    
    axs[1].set_ylabel("throughput (Gbps)")
    axs[1].set_title("write")
    axs[1].set_xlabel("time bin with size {} ns".format(int(bucket_size)))
    # axs[1].set_yscale("log")

    figure.set_size_inches(12, 8)
    plt.legend()
    plt.savefig(os.path.join(log_folder, "tpt.png"))
    return trace_dirt

def plot_runtime_arrival_rate(log_folder, inner_key_func=lambda f: True, outer_key_func=lambda f: False, bucket_size=1e6):
    figure, axs = plt.subplots(2)
    trace_files = get_trace_files(log_folder, inner_key_func, outer_key_func)
    for f in trace_files:
        df = pd.read_csv(f)
        df = df.sort_values(by="ArrivalTime", ignore_index=True)
        df["time_ms"] = (df.ArrivalTime/bucket_size).astype(int) + 1 # bucket with size of 1 ms
        r_df=df[df.IOType==0]
        w_df=df[df.IOType==1]
        r_rate = r_df.groupby(["time_ms"]).count()
        w_rate = w_df.groupby(["time_ms"]).count()
        marker = ''
        linestyle = "solid"
        axs[0].plot(r_rate.ArrivalTime, label=os.path.basename(f).split(".")[0], marker=marker, linestyle=linestyle)
        axs[1].plot(w_rate.ArrivalTime, label=os.path.basename(f).split(".")[0], marker=marker, linestyle=linestyle)
        break
    axs[0].set_ylabel("Arrival rate")
    axs[0].set_title("read")
    
    axs[1].set_ylabel("Arrival rate")
    axs[1].set_title("write")
    axs[1].set_xlabel("time bin with size {} ns".format(int(bucket_size)))
    # axs[1].set_yscale("log")
    figure.set_size_inches(12, 8)
    plt.legend()
    plt.savefig(os.path.join(log_folder, "rate.png"))

def plot_throughput_summary(log_folder, inner_key_func=lambda f: True, outer_key_func=lambda f: False, bucket_size=1e6, quantile=0.5, scale="linear"):
    # bar plot shows the throughput of each experiment 
    figure, axs = plt.subplots(2)
    tpt_df = pd.DataFrame(columns=["f", "r", "w"])
    trace_files = get_trace_files(log_folder, inner_key_func, outer_key_func)
    for f in trace_files:
        df = pd.read_csv(f)
        df = df.sort_values(by="FinishTime", ignore_index=True)
        # df["runtime_throughput"] = df.Size * 8 / (1024*1024*1024) # Gb 
        df["time_ms"] = (df.FinishTime/bucket_size).astype(int) + 1 # bucket with size of 1 ms
        r_df=df[df.IOType==0]
        w_df=df[df.IOType==1]
        r_tpt = r_df.groupby(["time_ms"]).sum()
        w_tpt = w_df.groupby(["time_ms"]).sum()
        r = (r_tpt.Size * 8 / (1024*1024*1024) / (bucket_size * 1e-9)).quantile(quantile)
        w = (w_tpt.Size * 8 / (1024*1024*1024) / (bucket_size * 1e-9)).quantile(quantile)
        tpt_df.loc[len(tpt_df), :] = [os.path.basename(f)[:-12], r, w]
        marker = '*'
        linestyle = "solid"
        axs[0].bar(tpt_df.f, tpt_df.r, label=os.path.basename(f).split(".")[0],)
        axs[1].bar(tpt_df.f, tpt_df.w,label=os.path.basename(f).split(".")[0], )

    axs[0].set_ylabel("throughput (Gbps)")
    axs[0].set_title("read")
    axs[0].set_yscale(scale)
    
    axs[1].set_ylabel("throughput (Gbps)")
    axs[1].set_title("write")
    axs[1].set_xlabel("time bin with size {} ns".format(int(bucket_size)))
    axs[1].set_yscale(scale)

    figure.set_size_inches(12, 8)
    plt.legend()
    plt.savefig(os.path.join(log_folder, "tpt_summary.png"))
    return tpt_df


def plot_runtime_tpt_v2(log_folder, inner_key_func=lambda f: True, outer_key_func=lambda f: False, bucket_size=1e6, quantile=0.3):
    figure, axs = plt.subplots(2)
    tpt_df = pd.DataFrame(columns=['f', 'r', 'w'])
    runtime_tpt_df = pd.DataFrame(columns=['time', 'r', 'w'])
    trace_files = get_trace_files(log_folder, inner_key_func, outer_key_func)
    for f in trace_files:
        df = pd.read_csv(f)
        for idx in df.index:
            win_start = int(df.loc[idx, "ArrivalTime"].item()/bucket_size)
            win_finish = int(df.loc[idx, "FinishTime"].item()/bucket_size)+1
            ave_tpt = df.loc[idx, "Size"].item()/(win_finish-win_start)
            for time_window in range(win_start, win_finish):
                if df.loc[idx, "IOType"].item()==0:
                    runtime_tpt_df.loc[len(runtime_tpt_df), :] = [time_window, ave_tpt, 0]
                else:
                    runtime_tpt_df.loc[len(runtime_tpt_df), :] = [time_window, 0, ave_tpt]

        sum_runtime_tpt_df = runtime_tpt_df.groupby(["time"]).sum()
        r = sum_runtime_tpt_df.r.quantile(quantile)
        w = sum_runtime_tpt_df.w.quantile(quantile)

        tpt_df.loc[len(tpt_df), :] = [os.path.basename(f)[:-12], r, w]
        marker = '*'
        linestyle = "solid"
        axs[0].bar(tpt_df.index, tpt_df.r, label=os.path.basename(f).split(".")[0],)
        axs[1].bar(tpt_df.index, tpt_df.w, label=os.path.basename(f).split(".")[0], )


    axs[0].set_ylabel("throughput (Gbps)")
    axs[0].set_title("read")
    # axs[0].set_yscale("log")
    
    axs[1].set_ylabel("throughput (Gbps)")
    axs[1].set_title("write")
    axs[1].set_xlabel("time bin with size {} ns".format(int(bucket_size)))
    # axs[1].set_yscale("log")

    figure.set_size_inches(12, 8)
    plt.legend()
    plt.savefig(os.path.join(log_folder, "run_time_tpt.png"))
    return tpt_df

def plot_runtime_onservice_rate(log_folder, inner_key_func=lambda f: True, outer_key_func=lambda f: False, bucket_size=1e6):
    figure, axs = plt.subplots(3)
    trace_files = get_trace_files(log_folder, inner_key_func, outer_key_func)
    for f in trace_files:
        with open(f, "r") as scheduler_trace:
            lines = scheduler_trace.read().splitlines()
        trace_lines = [l.split(",") for l in lines if len(l.split(","))==3]
        trace_df = pd.DataFrame(trace_lines, columns=['IOType','idx', 'onservicetime'])
        # plot the on service rate for w/r
        trace_df.loc[:, 'onservicetime'] = trace_df.onservicetime.astype(int)
        trace_df.loc[:, 'idx'] = trace_df.idx.astype(int)
        df = trace_df.sort_values(by="onservicetime", ignore_index=True)
        df["time_ms"] = (df.onservicetime/bucket_size).astype(int) + 1 # bucket with size of 1 ms
        r_df=df[df.IOType=='read']
        w_df=df[df.IOType=='write']
        r_rate = r_df.groupby(["time_ms"]).count()
        w_rate = w_df.groupby(["time_ms"]).count()
        r_w_rate_ratio = r_rate.idx/w_rate.idx
        marker = ''
        linestyle = "solid"
        axs[0].plot(r_rate.idx, label=f.split("/")[-2], marker=marker, linestyle=linestyle)
        axs[1].plot(w_rate.idx, label=f.split("/")[-2], marker=marker, linestyle=linestyle)
        # plot the number of requets on servive for w/r
        axs[2].plot(r_w_rate_ratio, label=f.split("/")[-2], marker=marker, linestyle=linestyle)

    axs[0].set_ylabel("on service rate")
    axs[0].set_title("read")

    axs[1].set_ylabel("on service rate")
    axs[1].set_title("write")

    axs[2].set_ylabel("read/write ratio")
    axs[2].set_title("ratio")
    axs[2].set_yscale('log')
    # labels = [eval(item.get_text()) for item in axs[2].get_xticklabels()]
    # axs[2].set_xticklabels([ str(int(item*bucket_size/1e6)) for item in labels])

    axs[2].set_xlabel("time bin with size {} ns".format(int(bucket_size)))
    figure.set_size_inches(12, 8)
    plt.legend()
    plt.savefig(os.path.join(log_folder, "onservice_rate.png"))
    return trace_df


def get_tpt_matrix_ratio(tpt_df):
    name_dict = {"1:1RW":"50%", "4:1RW":"80%", "1:4RW":"20%", "W":"0%", "R":"100%", 
                    "3:1RW":"75%", "1:3RW":"25%", "2:3RW":"40%", "3:2RW":"60%",
                    "1:9RW":"10%", "9:1RW":"90%", "1:2RW":"30%", "2:1RW":"70%"}
    tpt_df.to_csv(os.path.join(log_folder, "tpt_summary.csv"), index=False)
    tpt_df = pd.read_csv(os.path.join(log_folder, "tpt_summary.csv"))
    tpt_df["size"] = tpt_df.f.apply(lambda x: x.split("_")[2])
    tpt_df["interarrival"] = tpt_df.f.apply(lambda x: x.split("_")[1])
    tpt_df["ww"] = tpt_df.f.apply(lambda x: x.split("_")[-1][:-2]).astype(int)
    tpt_df["ratio"] = tpt_df.f.apply(lambda x: name_dict[x.split("_")[0]])
    names = tpt_df["ratio"].drop_duplicates().values
    names = sorted(names)
    arrival = tpt_df["interarrival"].drop_duplicates().values
    arrival = sorted(arrival, key=lambda x: int(x[:-2]))
    r_tpt = pd.DataFrame(columns = names, index = arrival)
    w_tpt = pd.DataFrame(columns = names, index = arrival)
    for i in tpt_df[tpt_df["ww"]=='1ww'].index:
        r_tpt.loc[tpt_df.loc[i, 'interarrival'], tpt_df.loc[i, 'ratio']] = tpt_df.loc[i, 'r']
        w_tpt.loc[tpt_df.loc[i, 'interarrival'], tpt_df.loc[i, 'ratio']] = tpt_df.loc[i, 'w']
    return tpt_df, r_tpt, w_tpt

def get_tpt_matrix(tpt_df):
    tpt_df.to_csv(os.path.join(log_folder, "tpt_summary.csv"), index=False)
    tpt_df = pd.read_csv(os.path.join(log_folder, "tpt_summary.csv"))
    tpt_df["size"] = tpt_df.f.apply(lambda x: x.split("_")[2])
    tpt_df["interarrival"] = tpt_df.f.apply(lambda x: x.split("_")[1])
    tpt_df["ww"] = tpt_df.f.apply(lambda x: x.split("_")[-1])

    names = tpt_df["size"].drop_duplicates().values
    names = sorted(names, key=lambda x: int(x[:-1]))
    arrival = tpt_df["interarrival"].drop_duplicates().values
    arrival = sorted(arrival, key=lambda x: int(x[:-2]))
    r_tpt = pd.DataFrame(columns = names, index = arrival)
    w_tpt = pd.DataFrame(columns = names, index = arrival)

    for i in tpt_df.index:
        r_tpt.loc[tpt_df.loc[i, 'interarrival'], tpt_df.loc[i, 'size']] = tpt_df.loc[i, 'r']
        w_tpt.loc[tpt_df.loc[i, 'interarrival'], tpt_df.loc[i, 'size']] = tpt_df.loc[i, 'w']

    return tpt_df, r_tpt, w_tpt

def plot_tpt_across_weights(tpt_df):
    tpt_df, r_tpt, w_tpt = get_tpt_matrix(tpt_df)
    matplotlib.rcParams.update({'font.size': 16, 'font.weight':'normal'})
    figure, axs = plt.subplots(len(r_tpt.index.values),len(r_tpt.keys().values))
    for i, size in enumerate(r_tpt.keys().values):
        for j, inter_arrival in enumerate(r_tpt.index.values):
            y_r = tpt_df.loc[(tpt_df["size"]==size)&(tpt_df["interarrival"]==inter_arrival), ["r", "ww"]]
            y_w = tpt_df.loc[(tpt_df["size"]==size)&(tpt_df["interarrival"]==inter_arrival), ["w", "ww"]]
            axs[j, i].plot(y_r["ww"], y_r["r"], marker="o", label="read")
            axs[j, i].plot(y_w["ww"], y_w["w"], marker="*", label="write")
            axs[j, i].set_yscale("log")

    for i, size in enumerate(r_tpt.keys().values):
        axs[0, i].set_title(size)
    for j, inter_arrival in enumerate(r_tpt.index.values):
        axs[j, 0].set_ylabel(inter_arrival)

    plt.legend()
    figure.set_size_inches(32, 16)
    plt.savefig("tpt_summary.png", dpi=100)

def plot_tpt_across_weights_ratio(tpt_df):
    tpt_df, r_tpt, w_tpt = get_tpt_matrix_ratio(tpt_df)
    matplotlib.rcParams.update({'font.size': 16, 'font.weight':'normal'})
    figure, axs = plt.subplots(len(r_tpt.index.values),len(r_tpt.keys().values))
    for i, ratio in enumerate(r_tpt.keys().values):
        for j, inter_arrival in enumerate(r_tpt.index.values):
            y_r = tpt_df.loc[(tpt_df["ratio"]==ratio)&(tpt_df["interarrival"]==inter_arrival), ["r", "ww"]]
            y_w = tpt_df.loc[(tpt_df["ratio"]==ratio)&(tpt_df["interarrival"]==inter_arrival), ["w", "ww"]]
            y_r = y_r.sort_values(by=["ww"])
            y_w = y_w.sort_values(by=["ww"])
            axs[j, i].plot(y_r["ww"].astype(str), y_r["r"], marker="o", label="read")
            axs[j, i].plot(y_w["ww"].astype(str), y_w["w"], marker="*", label="write")
            axs[j, i].set_yscale("log")
    for i, ratio in enumerate(r_tpt.keys().values):
        axs[0, i].set_title(ratio)
    for j, inter_arrival in enumerate(r_tpt.index.values):
        axs[j, 0].set_ylabel(inter_arrival)
    plt.legend()
    figure.set_size_inches(24, 16)
    plt.savefig("tpt_summary.png", dpi=100)

if __name__ == "__main__":
    bucket_size=1e6/10
    log_folder='/home/labuser/Downloads/MQSim/qmap_logs/Fujitsu_time_traces/Fujitsu_V0_based_10us_10_to_1_net-ssd'
    # ret_df = get_statistic_df(log_folder)

    # plot_runtime_arrival_rate(log_folder, bucket_size=1e6)
    # trace_dirt = plot_runtime_tpt_v2(log_folder, bucket_size=bucket_size)
    # plot_runtime_onservice_rate(log_folder, bucket_size=bucket_size)
    interarrrival_inner_key_func = lambda x: len(x.split("_"))<=8 and \
                                    np.all([key in x for key in ["results.csv", "1:1RW", "1ww", "8192B", "50000req"]])
    size_inner_key_func = lambda x: len(x.split("_"))<=8 and \
                                    np.all([key in x for key in ["results.csv", "1:1RW", "1ww", "50000"]])
    inner_key_func =lambda x: np.all([key in x for key in ["results.csv"]])
    outer_key_func=lambda f: f.startswith("W") or f.startswith("R") or ("1:3RW" in f) or ("3:1RW" in f)
    tpt_df = plot_throughput_summary(log_folder, inner_key_func, outer_key_func, bucket_size=bucket_size, quantile=0.5)
    plot_runtime_throughput(log_folder, inner_key_func, outer_key_func, bucket_size=bucket_size)

    # a, r_tpt, w_tpt = get_tpt_matrix(tpt_df)
    # plot_tpt_across_weights_ratio(tpt_df)

    # wait ratio plot
    # trace_df = trace_df.sort_values(by='idx', ignore_index=True)
    # response_df = response_df.sort_values(by='ArrivalTime', ignore_index=True)
    # read_response_df = response_df[response_df.IOType==0].reset_index(drop=True)
    # write_response_df = response_df[response_df.IOType==1].reset_index(drop=True)
    # read_onservice_df = trace_df[trace_df.IOType=='read'].reset_index(drop=True)
    # write_onservice_df = trace_df[trace_df.IOType=='write'].reset_index(drop=True)

    # waiting_df = read_onservice_df.onservicetime - read_response_df.ArrivalTime
    # waiting_ratio = waiting_df/read_response_df.DelayTime 
    # y = waiting_ratio[(waiting_ratio>0)&(waiting_ratio<1)]
    # figure, axs = plt.subplots(2)
    # axs[0].scatter( y.index, y )
    # axs[0].set_title("read")
    # axs[0].set_ylabel("waiting ratio")

    # waiting_df = write_onservice_df.onservicetime - write_response_df.ArrivalTime
    # waiting_ratio = waiting_df/write_response_df.DelayTime
    # y = waiting_ratio[(waiting_ratio>0)&(waiting_ratio<1)]
    # axs[1].scatter(y.index, y)
    # axs[1].set_title("write")
    # axs[1].set_ylabel("waiting ratio")
    # axs[1].set_xlabel("request index")
    # figure.set_size_inches(12, 8)
    # plt.savefig(os.path.join(log_folder, "wait_ratio.png"))