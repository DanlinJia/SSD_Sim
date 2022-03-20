import imp
import os
import multiprocessing
import subprocess
import random
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from xmanager import *
from bisect import bisect_left

def hyper_expo(phase_pro_vector, rate_vector):
    assert(len(phase_pro_vector)==len(rate_vector))
    assert(np.array(phase_pro_vector).sum()==1.0)
    phase_cumulative_vector = [ np.array(phase_pro_vector[:i+1]).sum() for i, pro in enumerate(phase_pro_vector)]
    phase_rnd = random.uniform(0, 1)
    i_phase = bisect_left(phase_cumulative_vector, phase_rnd)
    return random.expovariate(rate_vector[i_phase])

def number_generator(dist_args: dict):
    if 'dist_name' not in dist_args:
        raise()
    if dist_args['dist_name'] == 'uniform':
        a = dist_args['low']
        b = dist_args['high']
        return random.uniform(a, b)
    elif dist_args['dist_name'] == 'expo':
        mean = dist_args['mean']
        return random.expovariate(1/mean)
    elif dist_args['dist_name'] == 'gamma':
        alpha = dist_args['alpha']
        beta = dist_args['beta']
        return random.gammavariate(alpha, beta)
    elif dist_args['dist_name'] == 'gauss':
        mu = dist_args['mu']
        sigma = dist_args['sigma']
        return random.gauss(mu, sigma)
    elif dist_args['dist_name'] == 'hyper':
        phase_pro_vector, rate_vector = dist_args['phase_vector'], dist_args['rate_vector']
        return hyper_expo(phase_pro_vector, rate_vector)

def number_generator_test():
    n = 4
    dist_args = [0]*100
    fig, axs = plt.subplots(n)
    dist_args[0] = {'dist_name':'uniform', 'low':5, 'high':15}
    dist_args[1] = {'dist_name':'expo', 'mean':10}
    dist_args[2] = {'dist_name':'gamma', 'alpha':10, 'beta':1}
    # dist_args[3] = {'dist_name':'guass', 'mu':10, 'sigma':1}
    # Mean (SD): 9.7e+03 (2.2e+04) seconds.
    size_rate = 1/120
    ratio_list = []
    for a in np.arange(0.1, 1.95 ,0.05):
        ratio_list.append([0.05, 0.05, 2.90])
    for i, r in enumerate(ratio_list):
        dist_args[i] = {'dist_name':'hyper', 'phase_vector':[1/3, 1/3, 1/3], 'rate_vector':[1/r[0]*size_rate, 1/r[1]*size_rate, 1/r[2]*size_rate]}
    # dist_args[3] = {'dist_name':'hyper', 'phase_vector':[1/3, 1/3, 1/3], 'rate_vector':[3*size_rate, 3/2*size_rate, 1/2*size_rate]}

    try:
        req = 100000
        for i, a in enumerate(dist_args):
            x = []
            for j in range(req):
                x.append(number_generator(a))
            x = np.array(x)
            print("{}, mean: {}, std: {}, scv: {}".format(a['dist_name'], x.mean(), x.std(), pow(x.std()/x.mean(), 2)))
            # axs[i].scatter(list(range(req)), x, label=a['dist_name'])
    except Exception as e:
        print(e)
    plt.legend()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('test.png')


def generator(interarrival_mean, size, num, iotype, volume_id=0, target_id=0, hyper=False):
    # generate a df for a certain type of IOs
    df = pd.DataFrame(columns=["ArrivalTime", "VolumeID", "Offset", "Size", "IOType"])
    df["Offset"] = np.random.uniform(low=0, high=10*1024*1024*1024, size=num).astype(int)
    df["VolumeID"] = np.array([volume_id]*num)
    df["InitiatorID"] = df["VolumeID"]
    df["TargetID"] = np.array([target_id]*num)
    df["IOType"] = np.array([iotype]*num)
    if hyper:
        size_rate = 1/eu.Byte(size).num
        rate_ratio = [0.05, 0.05, 2.90]
        size_dist_args = {'dist_name':'hyper', 'phase_vector':[1/3, 1/3, 1/3], 'rate_vector':[1/rate_ratio[0]*size_rate, 1/rate_ratio[1]*size_rate, 1/rate_ratio[2]*size_rate]}
        interarrival_dist_args = {'dist_name':'hyper', 'phase_vector':[1/3, 1/3, 1/3], 'rate_vector':[1/eu.ns(interarrival_mean).num, 1/eu.ns(interarrival_mean).num, 1/eu.ns(interarrival_mean).num]}
    else:
        size_dist_args = {'dist_name':'expo', 'mean': eu.Byte(size).num}
        interarrival_dist_args = {'dist_name':'expo', 'mean':eu.ns(interarrival_mean).num}
    df["Size"] = (np.array([number_generator(size_dist_args)/512 for i in range(num)]).astype(int)+ 1)*512
    interarrival_list = np.array([number_generator(interarrival_dist_args) for i in range(num-1)])
    # interarrival_list = np.array([number_generator(interarrival_dist_args) for i in range(num)])
    df["ArrivalTime"] = (np.insert(np.cumsum(interarrival_list), 0, 0)).astype(int)
    # fake data
    df["DelayTime"] = df["ArrivalTime"]
    df["FinishTime"] = df["ArrivalTime"]
    return df

def generate_trace(interarrival_mean, size, num, ratio, trace_path, read_size=0, write_size=0, hyper=False):
    #Type_of_Requests[1 for write, 0 for read]
    df = pd.DataFrame(columns=["ArrivalTime", "VolumeID", "Offset", "Size", "IOType"])
    num = num.num
    ratio = ratio.num
    num_read = int(num*ratio)
    num_write = num - num_read
    # generate traces with ns time unit and Byte size unit
    interarrival_mean = eu.ns(interarrival_mean)
    size = eu.Byte(size)
    if ratio==0:
        if read_size!=0:
            mix_df = generator(interarrival_mean, read_size, num_write, 1, hyper=hyper)
        else:
            mix_df = generator(interarrival_mean, size, num_write, 1, hyper=hyper)
    elif ratio==1:
        if write_size!=0:
            mix_df = generator(interarrival_mean, write_size, num_read, 0, hyper=hyper)
        else:
            mix_df = generator(interarrival_mean, size, num_read, 0, hyper=hyper)
    else:
        if write_size==0:
            read_df = generator(interarrival_mean, size, num_read, 0, hyper=hyper)
            write_df = generator(interarrival_mean/(1/ratio-1), size, num_write, 1, hyper=hyper)
        else:
            read_df = generator(interarrival_mean, read_size, num_read, 0, hyper=hyper)
            write_df = generator(interarrival_mean/(1/ratio-1), write_size, num_write, 1, hyper=hyper)
        mix_df = pd.concat([read_df, write_df])
        read_df["RequestID"] = np.array(range(len(read_df)))
        write_df["RequestID"] = np.array(range(len(write_df)))
    mix_df = mix_df.sort_values(by=["ArrivalTime"])
    mix_df.index = range(len(mix_df))
    mix_df["RequestID"] = np.array(range(len(mix_df)))
    if not os.path.exists(trace_path.parent):
        os.system("mkdir -p {}".format(trace_path.parent))
    mix_df.to_csv(trace_path, header=True, index=False, sep=",")

if __name__ == "__main__":
    # unit test
    eu = experiment_unit()
    trace_attri_dict = {
        "ratio": eu.ratio(0.5),
        "inter_arrival": eu.us(1),
        "size": eu.Byte(512),
        "req": eu.req_num(50000),
        "rw": eu.rw(1),
        "ww": eu.ww(1)
    }
    ep = xmanager(trace_attri_dict, {}, pl.Path("log"), pl.Path("trace"), "test")
    
    def generate_trace_test(ep: xmanager, over_write: bool):
        # generate workload if not existing
        trace_name = ep.input_folder/ep.trace_name.with_suffix(".csv")
        interarrival_mean = ep.attri["inter_arrival"].num #s
        size = ep.attri["size"].num
        n = ep.attri["req"].num
        ratio = ep.attri["ratio"].num
        if (not ep.input_folder.exists()) or over_write:
            generate_trace(interarrival_mean, int(size), n, ratio, trace_name)
    
    # generate_trace_test(ep, False)
    number_generator_test()