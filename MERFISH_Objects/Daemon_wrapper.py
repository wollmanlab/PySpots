from MERFISH_Objects.Image import *
from MERFISH_Objects.Daemons import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Hybe import *
from MERFISH_Objects.Deconvolution import *
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Position import *
from MERFISH_Objects.Dataset import *
from MERFISH_Objects.Classify import *
from MERFISH_Objects.Segment import *
from MERFISH_Objects.FISHData import *
import dill as pickle
import argparse
import shutil
import importlib
import os
import time
import sys
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", type=str, help="dtype.")
#     parser.add_argument("cword_config", type=str, help="Name of config file.")
    parser.add_argument("-d", "--daemon_path", type=str, dest="daemon_path", default='/scratch/daemon', action='store', help="daemon_path")
    parser.add_argument("-n", "--ncpu", type=int, dest="ncpu", default=1, action='store', help="Number of cores")
    parser.add_argument("-i", "--interval", type=int, dest="interval", default=1, action='store', help="wait time")
    parser.add_argument("-v", "--verbose", type=bool, dest="verbose", default=True, action='store', help="loading bar")
    parser.add_argument("-ve", "--error_verbose", type=bool, dest="error_verbose", default=True, action='store', help="print errors")
    args = parser.parse_args()
        
if __name__ == '__main__':
#     merfish_config = importlib.import_module(args.cword_config)
    daemon_path = args.daemon_path
    if not os.path.exists:
        os.mkdir(daemon_path)
    dtype_daemon_path = os.path.join(daemon_path,args.dtype)
    if args.dtype =='deconvolution':
        dtype_daemon = Decovolution_Daemon(dtype_daemon_path,
                                           interval=args.interval,
                                           ncpu=args.ncpu,
                                           verbose=args.verbose,
                                           error_verbose = args.error_verbose)
    else:
        dtype_daemon = Class_Daemon(dtype_daemon_path,
                                    interval=args.interval,
                                    ncpu=args.ncpu,
                                    verbose=args.verbose,
                                    error_verbose = args.error_verbose)
    while True:
        time.sleep(100)
    