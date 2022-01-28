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
    parser.add_argument("metadata_path", type=str, help="dtype.")
    parser.add_argument("cword_config", type=str, help="Name of config file.")
    parser.add_argument("-i", "--interval", type=int, dest="interval", default=600, action='store', help="wait time")
    args = parser.parse_args()
        
def find_dtype(t):
    """ File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.csv' """
    """ dataset_positon_hybe_channel_zindex_flag.txt"""
    t = t.split('_')
    if t[-2]!='Zindex':
        if t[-4]=='Hybe':
            dtype = 'clas'
        else:
            dtype = 'img'
    elif t[-3]!='Channel':
        if t[-3]=='DeepBlue':
            if t[-4]=='Hybe':
                dtype='seg'
            else:
                dtype = 'reg'
        else:
            dtype = 'stk'
    elif t[-4]!='Hybe':
        dtype = 'hybe'
    elif t[-5]!='Position':
        dtype = 'pos'
    else:
        dtype = 'dataset'
    return dtype

def check_status(path,processed,dataset):
    out = {}
    dtypes = ['dataset','pos','seg','clas','hybe','reg','stk','img']
    outcomes = ['passed','failed','started']
    for dtype in dtypes:
        out[dtype] = {}
        for outcome in outcomes:
            out[dtype][outcome] = []
    for fname in os.listdir(path):
        if not dataset in fname:
            continue
        if 'flag' in fname:
            try:
                if fname in processed.keys():
                    t,dtype = processed[fname]
                else:
                    with open(os.path.join(path,fname),"r") as f:
                        t = f.read()
                        f.close()
                    dtype = find_dtype(fname)
                    if not 'Started' in t:
                        processed[fname] = (t,dtype)
            except:
                continue
            if 'Started' in t:
                out[dtype]['started'].append(fname)
            elif 'Passed' in t:
                out[dtype]['passed'].append(fname)
            elif 'Failed' in t:
                out[dtype]['failed'].append(fname)
                
    master_string = []
    for dtype in dtypes:
        string = dtype+'('
        for outcome in outcomes:
            string = string+str(len(out[dtype][outcome]))+':'
        string = string[:-1]+')'
        master_string.append(string)
    p = ''.join(i for i in master_string)
    return p,processed

if __name__ == '__main__':
    merfish_config = importlib.import_module(args.cword_config)
    daemon_path = merfish_config.parameters['daemon_path']
    
    metadata_path = args.metadata_path
    if metadata_path[-1]=='/':
        dataset = metadata_path.split('/')[-2]
    else:
        dataset = metadata_path.split('/')[-1]
    dataset_daemon_path = os.path.join(daemon_path,'dataset')
    processed = {}
    while True:
        p,processed = check_status(merfish_config.parameters['utilities_path'],processed,dataset)
        sys.stdout.write('\r'+str(datetime.now().strftime("%H:%M:%S"))+' '+p)
        sys.stdout.flush()
        time.sleep(args.interval)