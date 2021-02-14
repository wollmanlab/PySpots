from MERFISH_Objects.Analyze import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="dtype.")
    parser.add_argument("cword_config", type=str, help="Name of config file.")
    parser.add_argument("-i", "--interval", type=int, dest="interval", default=1, action='store', help="wait time")
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
    return p

if __name__ == '__main__':
    merfish_config = importlib.import_module(args.cword_config)
    daemon_path = merfish_config.parameters['daemon_path']
    
    metadata_path = args.metadata_path
    dataset_daemon_path = os.path.join(daemon_path,'dataset')
    while True:
        p = check_status(os.path.join(metadata_path,'fishdata'))
        sys.stdout.write('\r'+str(datetime.now().strftime("%H:%M:%S"))+' '+p)
        sys.stdout.flush()
        time.sleep(5)