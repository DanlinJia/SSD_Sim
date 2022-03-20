import time
import os

from pyrfc3339 import generate
import units
import pathlib as pl

class xmanager():
    def __init__(self, trace_attri_dict:dict, log_attri_dict: dict,
                    log_folder: pl.Path, trace_folder:pl.Path, 
                    experiment_set) -> None:
        
        super().__init__()
        
        self.trace_folder = trace_folder
        self.log_folder = log_folder
        self.attri = trace_attri_dict
        self.experiment_set_str = ""
        self.trace_name = self._generate_name_str(self.attri)
        self.unit = experiment_unit()
        
        if type(experiment_set)==str:
        # if experiment_set is pre-defined as str type
            self.experiment_set_str = experiment_set
        # generate experiment_set from attributes
        else:
            self.experiment_set_str = self._generate_name_str(
                {key:trace_attri_dict[key] for key in self.attri if key in experiment_set})

        for key in trace_attri_dict:
            locals()[key] = trace_attri_dict[key]
            assert(type(trace_attri_dict[key])==units.quantity.Quantity)

        # generate log_name if log has different attri from trace
        if log_attri_dict=={}:
            log_attri_dict = trace_attri_dict
            self.log_name = self.trace_name
        else:
            self.log_name = self._generate_name_str(log_attri_dict)
        
        # define the input and output folder for this experiment
        self.output_folder = self.log_folder/self.experiment_set_str/self.trace_name
        self.input_folder = self.trace_folder/self.experiment_set_str

    def _generate_name_str(self, attri_dict):
        name_dict = {0.5:"1:1RW", 0.8:"4:1RW", 0.2:"1:4RW", 0:"W", 1:"R", 
                    0.75:"3:1RW", 0.25:"1:3RW", 0.4:"2:3RW", 0.6:"3:2RW",
                    0.1:"1:9RW", 0.9:"9:1RW", 0.3:"1:2RW", 0.7:"2:1RW"}
        name = ""
        for key in attri_dict:
            if key=="ratio":
                attri_str = name_dict[attri_dict[key].get_num()]
            else:
                attri_str = "{}{}".format(int(attri_dict[key].get_num()), attri_dict[key].get_unit())
            if name == "":
                name = attri_str
            else:
                name += "_"+attri_str
        return pl.PurePath(name)

class experiment_unit():
    def __init__(self) -> None:
        # size units
        self.bit = units.unit("bit")
        self.Byte = units.scaled_unit("B", "bit", 1024)
        self.KB = units.scaled_unit("KB", "B", 1024)
        self.MB = units.scaled_unit("MB", "KB", 1024)
        self.GB = units.scaled_unit("GB", "MB", 1024)
        # time units 
        self.ns = units.unit("ns")
        self.us = units.scaled_unit("us", "ns", 1e3)
        self.ms = units.scaled_unit("ms", "us", 1e3)
        self.s = units.scaled_unit("s", "ms", 1e3)
        self.m = units.scaled_unit("min", "s", 60)
        self.h = units.scaled_unit("h", "min", 60)
        # request number 
        self.req_num = units.unit("req")
        # SQ weights 
        self.rw = units.unit("rw")
        self.ww = units.unit("ww")
        # R/W ratio
        self.ratio = units.unit("RW")

eu = experiment_unit()
if __name__ == "__main__":
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
