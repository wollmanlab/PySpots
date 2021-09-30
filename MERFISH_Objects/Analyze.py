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
    parser.add_argument("metadata_path", type=str, help="Path to Dataset Directory.")
    parser.add_argument("cword_config", type=str, help="Name of config file.")
    args = parser.parse_args()

def find_dtype(t):
    """ dataset_positon_hybe_channel_zindex_flag.txt"""
    t = t.split('_')
    if t[-2]!='X':
        dtype = 'img'
    elif t[-3]!='X':
        if t[-3]=='DeepBlue':
            if t[-4]=='X':
                dtype='seg'
            else:
                dtype = 'reg'
        else:
            dtype = 'stk'
    elif t[-4]!='X':
        if t[-4]=='all':
            dtype = 'clas'
        else:
            dtype = 'hybe'
    elif t[-5]!='X':
        dtype = 'pos'
    else:
        dtype = 'dataset'
    return dtype

def check_status(fishdata_path):
    out = {}
    dtypes = ['dataset','pos','seg','clas','hybe','reg','stk','img']
    outcomes = ['passed','failed','started']
    for dtype in dtypes:
        out[dtype] = {}
        for outcome in outcomes:
            out[dtype][outcome] = []
    for fname in os.listdir(fishdata_path):
        if 'flag' in fname:
            try:
                with open(os.path.join(fishdata_path,fname),"r") as f:
                    t = f.read()
                    f.close()
                dtype = find_dtype(fname)
            except:
                continue
            if 'Started'==t:
                out[dtype]['started'].append(fname)
            elif 'Passed'==t:
                out[dtype]['passed'].append(fname)
            elif 'Failed'==t:
                out[dtype]['failed'].append(fname)
                
    master_string = []
    for dtype in dtypes:
        string = dtype+'('
        for outcome in outcomes:
            string = string+str(len(out[dtype][outcome]))+':'
        string = string[:-1]+')'
        master_string.append(string)
    p = ''.join(i for i in master_string)
    return p

        
if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['GOTO_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    
    merfish_config = importlib.import_module(cword_config)
    daemon_path = merfish_config.parameters['daemon_path']
    utilities_path = merfish_config.parameters['utilities_path']
    if not os.path.exists(daemon_path):
        os.mkdir(daemon_path)
        os.chmod(daemon_path, 0o777)
    if not os.path.exists(utilities_path):
        os.mkdir(utilities_path)
        os.chmod(utilities_path, 0o777)
    dtypes = []
    dataset_daemon_path = os.path.join(daemon_path,'dataset')
        
    if metadata_path[-1]=='/':
        dataset = metadata_path.split('/')[-2]
    else:
        dataset = metadata_path.split('/')[-1]
    fname = str(dataset+'.pkl')
    data = {'metadata_path':metadata_path,
            'dataset':dataset,
            'cword_config':cword_config,
            'level':'dataset',
            'verbose':False}
    if not os.path.exists(dataset_daemon_path):
        os.mkdir(dataset_daemon_path)
        os.chmod(dataset_daemon_path, 0o777)
    if not os.path.exists(os.path.join(dataset_daemon_path,'input')):
        os.mkdir(os.path.join(dataset_daemon_path,'input'))
        os.chmod(os.path.join(dataset_daemon_path,'input'), 0o777)
    pickle.dump(data,open(os.path.join(dataset_daemon_path,'input',fname),'wb'))
        
    fishdata_path = os.path.join(metadata_path,'fishdata')
    if not os.path.exists(fishdata_path):
        os.mkdir(fishdata_path)
        os.chmod(fishdata_path, 0o777)
    while not os.path.exists(os.path.join(dataset_daemon_path,'output',fname)):
        p = check_status(os.path.join(metadata_path,'fishdata'))
        sys.stdout.write('\r'+str(datetime.now().strftime("%H:%M:%S"))+' '+p+'                     ')
        sys.stdout.flush()
        time.sleep(5)
    