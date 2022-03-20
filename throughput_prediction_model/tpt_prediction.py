import pickle
import pandas as pd
import numpy as np
import math
import pathlib as pl
import graphviz
from xmanager import *
from io_sim_test import run_workload, set_weights
from get_depart_rate import get_trace_files
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm, tree
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def feature_extraction(trace_df, feature: str, u):
    """
    extract feastures from a trace dataframe.
    """
    if feature == "Interarrival":
        inter_arrival = np.diff(trace_df["ArrivalTime"])
        return u(inter_arrival.mean()), u(inter_arrival.std())
    else:
        if feature not in trace_df.keys():
            raise Exception("feature is non-eixstent in current trace {}".format(trace_df.__name__))
    return u(trace_df[feature].mean()), u(trace_df[feature].std())

def bandwidth_calculation(trace_df, iotype, bucket_size=1e6, quantile=0.5):
    io_df = trace_df[trace_df.IOType==iotype]
    io_bw = io_df.groupby(["time_window"]).sum()
    bw = (io_bw.Size * 8 / (1024*1024*1024) / (bucket_size * 1e-9)).quantile(quantile)
    return bw

def arrival_flow_speed(trace_df, targetid=0, bucket_size=1e6, quantile=0.5):
    trace_df = trace_df.sort_values(by="ArrivalTime", ignore_index=True)
    trace_df["time_window"] = (trace_df.ArrivalTime/bucket_size).astype(int) + 1
    r = bandwidth_calculation(trace_df, iotype=0, bucket_size=1e6, quantile=0.5)
    w = bandwidth_calculation(trace_df, iotype=1, bucket_size=1e6, quantile=0.5)
    return [r, w]

def tpt_calculation(trace_path: str, targetid=0, bucket_size=1e6, quantile=0.5):
    """
    calculate r/w throughput, defined as the size of data processed per bucket_size nanosecond. 
    we take a 0.5 qutantile of throughput array as a throughput measurement.
    """
    trace_df = pd.read_csv(trace_path, header=0)
    trace_df = trace_df.sort_values(by="FinishTime", ignore_index=True)
    trace_df["time_window"] = (trace_df.FinishTime/bucket_size).astype(int) + 1 # bucket with size of 1 ms
    r = bandwidth_calculation(trace_df, iotype=0, bucket_size=1e6, quantile=0.5)
    w = bandwidth_calculation(trace_df, iotype=1, bucket_size=1e6, quantile=0.5)
    return pd.DataFrame({("Read", "tpt", ""):r, ("Write", "tpt", ""):w }, index=[targetid])


def trace_analyzer(trace_path: str):
    """
    analyze traces to collect characteristics and features for tpt prediction model.
    a trace contains requests sent from n initiators to m targets.
    @this returns a feature dataframe containing r/w features for each target.
    a trace contains the following columns
    RequestID,ArrivalTime,DelayTime,FinishTime,InitiatorID,TargetID,IOType,Size,VolumeID,Offset
    For input traces, DelayTime and FinishTime are artificially inserted.
    """
    trace_df = pd.read_csv(trace_path, header=0)
    target_index = trace_df.TargetID.drop_duplicates().values
    feature_column = pd.MultiIndex.from_product([["Read", "Write"], ["Size", "Interarrival"], ["AVE", "SCV"]])
    feature_df = pd.DataFrame(index=target_index, columns=feature_column)
    feature_df.index.name = "TargetID"
    io_map = {0:"Read", 1:"Write"}
    for targetid in trace_df.TargetID.drop_duplicates().values:
        target_df = trace_df[trace_df.TargetID==targetid]
        # calculate features for read/write I/Os, 1 is write, 0 is read.
        for iotype in [0, 1]:
            try:
                io_df = target_df[target_df["IOType"]==iotype]
                io_df.name = '{}_{}'.format(trace_path, io_map[iotype])
                ave_size, std_size = feature_extraction(io_df, "Size", eu.Byte)
                ave_interarrival, std_interarrival = feature_extraction(io_df, "Interarrival", eu.ns)
                feature_df.loc[targetid, io_map[iotype] ] = [ave_size, math.pow(std_size/ave_size, 2), \
                                                            ave_interarrival, math.pow(std_interarrival/ave_interarrival, 2)]
            except RuntimeError:
                print(trace_path, targetid, iotype)
    # read_ratio = read_num/(read_num+write_num) = 1/(1+write_num/read_mum)
    feature_df["Read_Ratio"] = 1/(1+feature_df.Write.Interarrival.AVE/feature_df.Read.Interarrival.AVE)
    feature_df["trace_name"] = os.path.basename(trace_path)
    feature_df["R_W_Size_Ratio"] = feature_df.Read.Size.AVE/feature_df.Write.Size.AVE
    r_arrival_speed, w_arrival_speed = arrival_flow_speed(trace_df, targetid=0, bucket_size=1e6, quantile=0.5)
    feature_df[("Read", "ArrivalFlowSpeed", "")] = r_arrival_speed
    feature_df[("Write", "ArrivalFlowSpeed", "")] = w_arrival_speed
    return feature_df

def tpt_feature_engineering(trace_path: str, log_folder: str, save_feature=False, feature_name=""): 
    """
    run workload to collect the real throughput.
    """
    feature_df = trace_analyzer(trace_path)
    try:
        write_weight = int(pl.PurePath(trace_path).stem.split("_")[5][:-2])
    except:
        try:
            write_weight = int(pl.PurePath(trace_path).stem.split("_")[-2])
        except:
            write_weight = 1
    feature_df["ww"] = [write_weight] * len(feature_df)
    if save_feature:
        if feature_name=="":
            feature_name = pl.PurePath(trace_path).stem
        log_path = pl.Path(log_folder) / feature_name
        if not log_path.exists():
            log_path.mkdir()
        feature_df.to_csv((log_path/("feature_"+feature_name)).with_suffix(".csv"), index=True)
        # run_workload(trace_path, log_path)
    return feature_df

def data_preparation(dataset_folder: str, inner_key_func=lambda x:True, outer_key_func=lambda x:True):
    data_list = []
    trace_files = get_trace_files(dataset_folder, inner_key_func, outer_key_func)
    for trace_path in trace_files:
        # try:
        feature_df = tpt_feature_engineering(trace_path, "logs")
        for targetid in feature_df.index:
            tpt_df = tpt_calculation(trace_path, targetid)
            data_list.append(feature_df.join(tpt_df))
        # except Exception as e:
        #     print("error in processing trace {}".format(os.path.basename(trace_path)), e)
    return pd.concat(data_list), feature_df.keys(), tpt_df.keys()

def load_val_data(validation_folder):
    inner_key_func =lambda x: np.all([key in x for key in ["results.csv"]])
    outer_key_func=lambda f: f.startswith("W") or f.startswith("R") 
    val_df, train_key, val_key = data_preparation(validation_folder, inner_key_func, outer_key_func)
    if "trace_name" in train_key:
        train_key = train_key.drop("trace_name")
    val_x, val_y = val_df[train_key].values, val_df[val_key].values
    return val_df, train_key, val_key, val_x, val_y

def load_train_data(training_folder):
    inner_key_func =lambda x: np.all([key in x for key in ["results.csv", "1ww"]])
    outer_key_func=lambda f: f.startswith("W") or f.startswith("R")
    train_df, train_key, val_key = data_preparation(training_folder, inner_key_func, outer_key_func)
    if "trace_name" in train_key:
        train_key = train_key.drop("trace_name")
    train_x, train_y = train_df[train_key].values, train_df[val_key].values
    return train_df, train_key, val_key, train_x, train_y

def load_data(training_folder: str, validation_folder='', data_reload=False, data_save=False, data_shuffle=False, training_ratio=0.8):
    if validation_folder=='':
        if data_reload:
            dataset_folder = training_folder
            inner_key_func =lambda x: np.all([key in x for key in ["results.csv", "1ww"]])
            outer_key_func=lambda f: f.startswith("W") or f.startswith("R")
            data_df, train_key, val_key = data_preparation(dataset_folder, inner_key_func, outer_key_func)
            data_df = data_df.reset_index()
            train_df = data_df.sample(frac=training_ratio)
            val_df = data_df[~data_df.index.isin(train_df.index)]
            if data_save:
                with open(os.path.join("/home/labuser/Downloads/MQSim/tpt_dataset", "data_set.pickle"), "wb") as data_set:
                    pickle.dump((data_df, train_df, val_df, train_key, val_key), data_set, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join("/home/labuser/Downloads/MQSim/tpt_dataset", "data_set.pickle"), "rb") as data_set:
                data_df, train_df, val_df, train_key, val_key = pickle.load(data_set)
            if data_shuffle:
                train_df = data_df.sample(frac=training_ratio)
                val_df = data_df[~data_df.index.isin(train_df.index)]
        if "trace_name" in train_key:
            train_key = train_key.drop("trace_name")
        train_x, train_y = train_df[train_key].values, train_df[val_key].values
        val_x, val_y = val_df[train_key].values, val_df[val_key].values
    
    else:
        val_df, train_key, val_key, val_x, val_y = load_val_data(validation_folder)
        train_df, train_key, val_key, train_x, train_y = load_train_data(training_folder)


    return val_df, train_key, val_key, val_x, val_y, train_df, train_x, train_y


def training_models(val_df, train_key, val_key, val_x, val_y, train_df, train_x, train_y, accuracy_fct=False):
    tpt_score_dict = {}
    tpt_lr_model = LinearRegression(fit_intercept=True, normalize=True).fit(train_x, train_y)
    transformer = PolynomialFeatures(degree=2, include_bias=True)
    x_ = transformer.fit_transform(train_x.astype(float).tolist())
    tpt_PLR_model = LinearRegression(fit_intercept=True, normalize=True).fit(x_, train_y)
    x_ = transformer.fit_transform(val_x.astype(float).tolist())
    tpt_knn_model = KNeighborsRegressor(n_neighbors=10).fit(train_x, train_y)
    tpt_rfr_model = RandomForestRegressor(max_depth=10, random_state=10).fit(train_x, train_y)
    tpt_mlp_model = MLPRegressor(random_state=12, max_iter=5000).fit(train_x, train_y)
    tpt_dtr_model =  DecisionTreeRegressor(max_depth=10).fit(train_x, train_y)
    if accuracy_fct:
        pre_y = tpt_lr_model.predict(val_x)
        tpt_score = accuracy_fct(pre_y, val_y)
        print("Linear Regression: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_lr_model
        pre_y = tpt_PLR_model.predict(x_)
        tpt_score = accuracy_fct(pre_y, val_y)
        print("Polynomial Regression: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_PLR_model
        pre_y = tpt_knn_model.predict(val_x)
        tpt_score = accuracy_fct(pre_y, val_y)
        print("K-Nearest Neighbor: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_knn_model
        pre_y = tpt_rfr_model.predict(val_x)
        tpt_score = accuracy_fct(pre_y, val_y)
        print("Random Forest Regressor: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_rfr_model
        pre_y = tpt_mlp_model.predict(val_x)
        tpt_score = accuracy_fct(pre_y, val_y)
        print("MLP: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_mlp_model
        pre_y = tpt_dtr_model.predict(val_x)
        tpt_score = accuracy_fct(pre_y, val_y)
        print("Decision Tree Regression: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_dtr_model
    else:
        tpt_score = tpt_lr_model.score(val_x, val_y)
        print("Linear Regression: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_lr_model
        tpt_score = tpt_PLR_model.score(x_, val_y)
        print("Polynomial Regression: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_PLR_model
        tpt_score = tpt_knn_model.score(val_x, val_y)
        print("K-Nearest Neighbor: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_knn_model
        tpt_score = tpt_rfr_model.score(val_x, val_y)
        print("Random Forest Regressor: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_rfr_model
        tpt_score = tpt_mlp_model.score(val_x, val_y)
        print("MLP: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_mlp_model
        tpt_score = tpt_dtr_model.score(val_x, val_y)
        print("Decision Tree Regression: {}".format(tpt_score))
        tpt_score_dict[tpt_score] = tpt_dtr_model
    with open(os.path.join('trained_models','tpt_model'), 'wb') as tpt_model_saved:
        tpt_model = tpt_score_dict[sorted(tpt_score_dict.keys())[-1]]
        pickle.dump(tpt_model, tpt_model_saved)
    return train_df, val_df, train_key, val_key, tpt_lr_model, tpt_knn_model, tpt_rfr_model, tpt_dtr_model

def tpt_learning(training_folder: str, validation_folder='', data_reload=False, data_save=False, data_shuffle=False, training_ratio=0.8, accuracy_fct=None):
    """
    learn throughput of workloads from their features.
    """
    val_df, train_key, val_key, val_x, val_y, train_df, train_x, train_y = \
        load_data(training_folder, validation_folder, data_reload, data_save, data_shuffle, training_ratio)
    return training_models(val_df, train_key, val_key, val_x, val_y, train_df, train_x, train_y, accuracy_fct)

def train_model(model, val_df, train_df, train_key, val_key):
    val_x, val_y = val_df[train_key].values, val_df[val_key].values
    train_x, train_y = train_df[train_key].values, train_df[val_key].values
    model.fit(train_x, train_y)
    score = model.score(val_x, val_y)
    return score

def test_feature_importance(training_folder: str, model, validation_folder='', data_reload=False, data_save=False, data_shuffle=False, training_ratio=0.8, accuracy_fct=None):
    val_df, train_key, val_key, val_x, val_y, train_df, train_x, train_y = \
        load_data(training_folder, validation_folder, data_reload, data_save, data_shuffle, training_ratio)
    
    feature_factor_df = pd.DataFrame(columns=["feature", "model" ,"score"])
    feature_pairs = [
        [('Read', 'ArrivalFlowSpeed', ''), ('Write', 'ArrivalFlowSpeed', '')],
        [('R_W_Size_Ratio', '', ''), ('Read_Ratio', '', '')], 
        [('R_W_Size_Ratio', '', '')], 
        [('Read_Ratio', '', '')],
        [( 'Read', 'Size', 'AVE'), ('Write', 'Size', 'AVE')],
        [( 'Read', 'Size', 'SCV'), ('Write', 'Size', 'SCV')],
        [( 'Read', 'Interarrival', 'AVE'), ('Write', 'Interarrival', 'AVE')],
        [( 'Read', 'Interarrival', 'SCV'), ('Write', 'Interarrival', 'SCV')],
        [('ww', '', '')]
    ]
    for feature in feature_pairs:
        part_train_key = train_key.drop(feature)
        tpt_model = RandomForestRegressor(max_depth=10, random_state=10)
        score = train_model(tpt_model, val_df, train_df, part_train_key, val_key)
        feature_factor_df.loc[len(feature_factor_df), :] = [str(feature), 'RandomForestRegressor', score]
            
        score = train_model(tpt_model, val_df, train_df, train_key, val_key)
        feature_factor_df.loc[len(feature_factor_df), :] = ["full", 'RandomForestRegressor', score]

    return feature_factor_df


def predict_tpt(model_path, feature):
    """
    predict the workload's throughput.
    """
    model_path = pl.Path(model_path)
    if not model_path.exists():
        raise Exception("Wrong model path")
    with open(model_path, 'rb') as tpt_model_saved:
        tpt_model = pickle.load(tpt_model_saved)

    tpt_prediction = tpt_model.predict(feature)
    return tpt_prediction

if __name__ == "__main__":
    trace_path = "/home/labuser/Downloads/MQSim/test/Fujitsu-tQFS_V0_based_15us_1_to_1_net-ssd_ACF_ALL0.csv"
    feature_df = tpt_feature_engineering(trace_path, "/home/labuser/Downloads/MQSim/logs/50us_50000req_55000B_27500B", save_feature=False)
    # set_weights(1,1)
    # tpt_feature_engineering(trace_path, "logs")
    # tpt_df = tpt_calculation(trace_path)
    # dataset_folder = "/home/labuser/Downloads/MQSim/tpt_dataset"
    # train_df, val_df = tpt_learning(dataset_folder, data_reload=True, data_save=True, data_shuffle=True, training_ratio=0.9)

    training_folder = "/home/labuser/Downloads/MQSim/logs"
    validation_folder = "/home/labuser/Downloads/MQSim/qmap_logs/Fujitsu_size_traces"
    # train_df, val_df, train_key, val_key, tpt_lr_model, tpt_knn_model, tpt_rfr_model, tpt_dtr_model = tpt_learning(training_folder, data_reload=True, data_save=True, data_shuffle=True, training_ratio=0.6)
    train_df, val_df, train_key, val_key, tpt_lr_model, tpt_knn_model, tpt_rfr_model, tpt_dct_model= tpt_learning(training_folder,validation_folder, \
         data_reload=True, data_save=True, data_shuffle=True, training_ratio=0.6, accuracy_fct=None)

def plt_dct(tpt_dct_model):
    fig = plt.figure(figsize=(28,12))
    _ = tree.plot_tree(tpt_dct_model, feature_names=train_key, filled=True, fontsize=10)
    plt.savefig("dct.png", dpi=100)

def train_df_analysis(train_df):
    train_df.Read.Interarrival.AVE.apply(eu.us).apply(lambda x: int(x.num)).value_counts()
    train_df[(train_df.Read.Interarrival.AVE.apply(eu.us)<eu.us(41))&(train_df.Write.Interarrival.AVE.apply(eu.us)>eu.us(39))]

def validate_on_folder(validation_folder, train_df):
    val_df, train_key, val_key, val_x, val_y = load_val_data(validation_folder)
    train_df, val_df, train_key, val_key, tpt_lr_model, tpt_knn_model, tpt_rfr_model, tpt_dct_model =  \
        training_models(val_df, train_key, val_key, val_x, val_y, train_df, train_x, train_y, accuracy_fct=False)