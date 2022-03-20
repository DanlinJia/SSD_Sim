import numpy as np
from synthetic_generator_v2 import *
from get_depart_rate import get_trace_files

qmap_folder = ""
inner_key_func=lambda x: ".txt" in x
qmap_files = get_trace_files(qmap_folder, inner_key_func)

parser = argparse.ArgumentParser(description="generate synthetic traces from QMAPs.")
parser.add_argument("--initiator_num", "-i", type=int, default=1, help="the number of initiators.")
parser.add_argument("--target_num", "-t", type=int, default=1, help="the number of targets.")
parser.add_argument("--request_num", "-n", type=int, default=50000, help="the number of requests to generate.")
parser.add_argument("--r_w_ratio", "-r", type=float, default=1.0, help="read write ratio.")
parser.add_argument("--sample_folder", "-sf", type=str, default='r_4kb_2us_3scv_w_2kb_3us_1scv', \
                    help="the folder where samples are saved.")
parser.add_argument("--trace_name", "-tn", type=str, default='r_4kb_2us_3scv_w_2kb_3us_1scv', \
                    help="the final generated trace name.")
parser.add_argument("--workspace", "-w", type=str, default='workspace', help="the folder contains QMAPs.")
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

# reset maps name, trace name, sample folder
# args.sample_folder = ''
# args.trace_name = ''
# args.read_arrival_time = ''
# args.read_size = ''
# args.write_arrival_time = ''
# args.write_size = ''

trace_df = trace_generation_from_qmaps(args)







