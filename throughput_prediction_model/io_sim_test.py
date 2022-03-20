# from ssd_simulator import *
from io_sim import *
from xmanager import *
import argparse
import pathlib
from micro_generator import *
from get_depart_rate import get_trace_files

def run_workload(workload, log_folder):
    ori = pd.read_csv(workload)
    # We use unique arrival time to identify requests
    ori = ori.drop_duplicates(subset=["ArrivalTime"])
    init_arrival = ori.ArrivalTime
    ssd_output = "net-ssd.csv"
    ssd_sim = ssd_simulator("./tmp",
                            workload, ".",
                            ssd_output)
    df = ssd_sim.ssd_simulation_iter(init_arrival)

    if not os.path.exists(log_folder):
        os.system("mkdir -p {}".format(log_folder))
    path = pathlib.PurePath(log_folder)
    df.to_csv(os.path.join(log_folder, "{}_results.csv".format(os.path.basename(log_folder))))
    os.system("./clear_log.sh {} {}".format(log_folder, "."))
    shutil.copy("/home/labuser/Downloads/MQSim/workload.trace_scenario_1.xml", log_folder)

    real_df = df 
    # real_df = df.sort_values(by="FinishTime")
    # real_df = real_df.loc[int(len(real_df)*1/8):int(len(real_df)*1/8),:]
    r_df=real_df[real_df.IOType==0]
    w_df=real_df[real_df.IOType==1]
    print("read delay: {}, write delay: {}".format(r_df.DelayTime.mean()/1e3, w_df.DelayTime.mean()/1e3))

def ssd_simulation_throughput(df, bin_num):
   
    interval_time = df["FinishTime"].max() -  df["FinishTime"].min()  
    out, bins = pd.cut(df["FinishTime"], bin_num, retbins=True, labels=False)
    size_sum = []
    for i in range(bin_num + 1):
        size_sum.append(df[out == i]["Size"].sum()/(interval_time/bin_num))
    return size_sum

def cdfplot(data, bin_num):
    counts, bin_edges= np.histogram(data, bins=bin_num, normed=False)
    cdf = np.cumsum(counts)
    return bin_edges[1:], cdf / cdf[-1]

def set_weights(r, w):
    config = "ssdconfig.test.xml"
    tree = et.parse(config)
    tree.find('Host_Parameter_Set/Read_Weights').text = str(r)
    tree.find('Host_Parameter_Set/Write_Weights').text = str(w)
    tree.write(config)

# def execute_workload(args):
#     if len(args.input_name)!=0:
#         experiment_name = args.input_name
#         output_folder = os.path.join("traces", args.output_name)
#         log_folder = os.path.join("logs", args.output_name)
#         run_workload(experiment_name, log_folder)
#     else:
#         interarrival_mean = args.interarrival #s
#         size = args.size
#         num = args.num
#         name_dict = {0.5:"1:1RW", 0.8:"4:1RW", 0.2:"1:4RW", 0:"W", 1:"R", 0.75:"3:1RW", 0.25:"1:3RW", 0.4:"2:3RW", 0.6:"3:2RW"}
#         trace_name = "{}_{}us_{}B_{}.csv".format(name_dict[args.ratio], int(interarrival_mean*1e6), int(size), num)
#         experiment_name = "{}us_{}B_{}".format(int(interarrival_mean*1e6), int(size), num)
#         # df = generator(interarrival_mean, size, num, iotype, volume_id=0, target_id=0)
#         output_folder = "traces/{}".format(experiment_name)
#         workload  = os.path.join(output_folder, trace_name)
#         if (not os.path.exists(workload)) or args.over_write:
#             generate_trace(interarrival_mean, int(size), num, output_folder, ratio=args.ratio, name=trace_name)
#         log_folder = os.path.join("logs", experiment_name, trace_name.split("_")[0], experiment_name+"_rw_{}_{}".format(args.read_weight, args.write_weight))
#         run_workload(workload, log_folder)

def execute_workload(ep: xmanager):
    # generate workload if not existing
    workload = ep.input_folder/ep.trace_name.with_suffix(".csv")
    interarrival_mean = ep.attri["inter_arrival"]
    size = ep.attri["size"]
    n = ep.attri["req"]
    ratio = ep.attri["ratio"]
    write_size = ep.attri.get("ws")
    if (not os.path.exists(workload)) or args.over_write:
        if write_size:
            generate_trace(interarrival_mean, int(size), n, ratio, workload, read_size=int(size), write_size=int(write_size), hyper=False)
        else:
            generate_trace(interarrival_mean, int(size), n, ratio, workload, hyper=False)
    run_workload(workload, ep.output_folder)

def usage():
    print(
        "mod 1 round_test generate micro traces and run simulation automatically \n \
        set parameters under 'if args.do_round_test:' \n \
        run with e.g., \n 'python -rt' ",
        "mod 2 dry_run runs simulation on exsiting traces \n \
        run with e.g., \n 'python -tf trace -lf logs -d' ",
        "mod 3 reads input arguments to generate a trace, following with a simulation. \
        run with e.g., \n 'python -a 0.000010 -s 4096 -n 50000 -r 0.5 -ww 1 '" 

    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_folder", "-tf", default="trace")
    parser.add_argument("--log_folder", "-lf", default="logs")
    parser.add_argument("--over_write", '-ow', dest='over_write', action='store_true')
    parser.add_argument("--interarrival", "-a", default=0, type=float)
    parser.add_argument("--size", "-s", default=0, type=float)
    parser.add_argument("--num", "-n", default=50000, type=int)
    parser.add_argument("--ratio", '-r', default=0.5, type=float)
    parser.add_argument("--read_weight", '-rw', default=1, type=int)
    parser.add_argument("--write_weight", '-ww', default=1, type=int)
    parser.add_argument("--round_test", '-rt', dest='do_round_test', action='store_true')
    parser.add_argument("--dry_run", '-d', dest='dry_run', action='store_true')
    args = parser.parse_args()

    set_weights(1, 1)
    if args.do_round_test:
        inter_arrival_list = [0.0001, 0.00008, 0.00004, 0.00002, 0.00001, 0.000008, 0.000006, 0.000004, 0.000002, 0.000001]
        # inter_arrival_list = [0.000004]
        size_list = [2048*5, 2048*10, 2048*15, 2048*20]
        # size = 8192
        ratio = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        write_weight = [1]
        for arrival in inter_arrival_list:
            for r in ratio:
                for size in size_list:
                    for sr in [0.2, 0.5, 0.8]:
                        args.interarrival = arrival
                        args.size = size
                        args.ratio = r
                        for ww in write_weight:
                            set_weights(1, ww)
                            args.write_weight = ww
                            trace_attri_dict = {
                                    "ratio": eu.ratio(args.ratio),
                                    "inter_arrival": eu.us(eu.s(args.interarrival)),
                                    "size": eu.Byte(args.size),
                                    "req": eu.req_num(args.num),
                                    "rw": eu.rw(args.read_weight),
                                    "ww": eu.ww(args.write_weight),
                                    "rs": eu.Byte(args.size),
                                    "ws": eu.Byte(args.size) * sr,
                            }
                            ep = xmanager(trace_attri_dict, {},
                                        pl.Path(args.log_folder), pl.Path(args.trace_folder),
                                        ["inter_arrival", "size", "req"])
                            try:
                                execute_workload(ep)
                            except Exception as e:
                                print(e, "\nhappens at arrival {} size {} for weights {}".format(arrival, size, [1, ww]))
    elif args.dry_run:
        inner_key_func=lambda x: ".csv" in x
        trace_files = get_trace_files(args.trace_folder, inner_key_func)
        for trace in trace_files:
            try:
                run_workload(trace, pl.Path(args.log_folder)/pl.Path(trace).stem)
            except Exception as e:
                print(e, trace)
    
    else:
        trace_attri_dict = {
                "ratio": eu.ratio(args.ratio),
                "inter_arrival": eu.us(eu.s(args.interarrival)),
                "size": eu.Byte(args.size),
                "req": eu.req_num(args.num),
                "rw": eu.rw(args.read_weight),
                "ww": eu.ww(args.write_weight),
                "rs": eu.Byte(args.size),
                "ws": eu.Byte(args.size) * 0.5,
        }
        set_weights(args.read_weight, args.write_weight)
        ep = xmanager(trace_attri_dict, {},
                                 pl.Path(args.log_folder), pl.Path(args.trace_folder),
                                 ["inter_arrival", "req", "rs" , "ws"])
        execute_workload(ep)


    # filename = "test"
    # workload = "/home/labuser/ssd-net-sim/traces/net-ssd-Test/"+ filename + ".csv"
    # workload = "/home/labuser/Downloads/MQSim/traces/test/mix_10us_4096B_100000.csv"
    


    # df_sum = 0
    # for i in range(len(df-1)):
    #     #df_sum = df_sum + (df.FinishTime.iloc[i] - df.ArrivalTime.iloc[i])#durantion 
    #     df_size = df.Size.sum()
    # df_sum = df.FinishTime.iloc[len(df)-1]-df.ArrivalTime.iloc[0]
    # print("overall throuput:" , (df_size/1e9)/(df_sum/1e9))

    # #write througput 
    # w_df_sum = 0
    # for i in range(len(w_df-1)):
    #     #w_df_sum = w_df_sum + (w_df.FinishTime.iloc[i] - w_df.ArrivalTime.iloc[i])
    #     w_df_size =  w_df.Size.sum()
    # w_df_sum = w_df.FinishTime.iloc[len(w_df)-1]-w_df.ArrivalTime.iloc[0]
    # print("write throuput:" , w_df_size/w_df_sum)   


    # #read througput 
    # r_df_sum = 0
    # for i in range(len(r_df-1)):
    #     #r_df_sum = r_df_sum + (r_df.FinishTime.iloc[i] - r_df.ArrivalTime.iloc[i])
    #     r_df_size = r_df.Size.sum()
    # r_df_sum = r_df.FinishTime.iloc[len(r_df)-1]-r_df.ArrivalTime.iloc[0]
    # print("read throuput:" , r_df_size/r_df_sum)
    
    # print ("AveDelay:", df.DelayTime.mean())
    # bin_num = 100
    # thr = ssd_simulation_throughput(ori, bin_num)
    # edges, cdf = cdfplot(thr, bin_num)
    
    # plt.figure()
    # plt.xlabel("Unit time ")
    # plt.ylabel("Throughput [Gbps]")
    # plt.plot(thr)
    # plt.savefig(filename + "_thr.png")
    # plt.figure()
    # plt.xlabel("Throughput [Gbps]")
    # plt.hist(thr)
    # plt.savefig(filename + "_pdf.png")
    # plt.figure()
    # #cdf
    # plt.xlabel("Throughput [Gbps]")
    # plt.ylabel("Probility")
    # plt.plot(edges, cdf)
    # plt.savefig(filename + "_cdf.png")
    