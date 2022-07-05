from fish_helpers import *
from MERFISH_Objects.FISHData import *
from tqdm import tqdm
""" Function to rerun class """
from MERFISH_Objects.Daemons import *
import parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="metadata_path")
    parser.add_argument("cword_config", type=str, help="Name of config file.")
    parser.add_argument("Type", type=str, help="Class to Run")
    parser.add_argument("-n", "--ncpu", type=int, dest="ncpu", default=30, action='store', help="Number of cores")
    parser.add_argument("-b", "--batches", type=int, dest="batches", default=2000, action='store', help="Number of batches")
    parser.add_argument("-v", "--verbose", type=bool, dest="verbose", default=False, action='store', help="loading bar")
    parser.add_argument("-r", "--rerun", type=bool, dest="rerun", default=False, action='store', help="rerun")
    args = parser.parse_args()
    print(args)

def generate_class(data):
    level = data['level']
    if 'verbose' in data.keys():
        verbose = data['verbose']
    else:
        verbose = False
    class_verbose = verbose
    if level == 'dataset':
        data_object = Dataset_Class(data['metadata_path'],
                                    data['dataset'],
                                    data['cword_config'],
                                    verbose=class_verbose)
    elif level == 'position':
        data_object = Position_Class(data['metadata_path'],
                                     data['dataset'],
                                     data['posname'],
                                     data['cword_config'],
                                     verbose=class_verbose)
    elif level == 'hybe':
        data_object = Hybe_Class(data['metadata_path'],
                                 data['dataset'],
                                 data['posname'],
                                 data['hybe'],
                                 data['cword_config'],
                                 verbose=class_verbose)
    elif level == 'registration':
        data_object = Registration_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['hybe'],
                                         data['cword_config'],
                                         verbose=class_verbose)
    elif level == 'stack':
        data_object = Stack_Class(data['metadata_path'],
                                  data['dataset'],
                                  data['posname'],
                                  data['hybe'],
                                  data['channel'],
                                  data['cword_config'],
                                  verbose=class_verbose)
    elif level == 'deconvolution':
        data_object = Deconvolution_Class(data['metadata_path'],
                                          data['dataset'],
                                          data['posname'],
                                          data['hybe'],
                                          data['channel'],
                                          data['cword_config'],
                                          verbose=class_verbose)
    elif level == 'image':
        data_object = Image_Class(data['metadata_path'],
                                  data['dataset'],
                                  data['posname'],
                                  data['hybe'],
                                  data['channel'],
                                  data['zindex'],
                                  data['cword_config'],
                                  verbose=class_verbose)
    elif level == 'segmentation':
        data_object = Segment_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=class_verbose)
    elif level == 'classification':
        data_object = Classify_Class(data['metadata_path'],
                                     data['dataset'],
                                     data['posname'],
                                     data['zindex'],
                                     data['cword_config'],
                                     verbose=class_verbose)
    return data_object

def manual_class(data):
    data_object = generate_class(data)
    data_object.main()
    return data
    # try:
    #     data_object.main()
    #     return data
    # except:
    #     print(data)
    #     return data   
    
if __name__ == '__main__':   
    metadata_path = args.metadata_path
    dataset = [i for i in metadata_path.split('/') if i!=''][-1]
    cword_config = args.cword_config
    config = importlib.import_module(cword_config)
    bitmap = config.bitmap
    Type = args.Type
    ncpu = args.ncpu
    verbose = args.verbose
    batches = args.batches
    rerun = args.rerun
    if Type=='registration':
        bitmap.append(['dapi','nucstain','DeepBlue'])
    # if Type=='classification':
    #     data = {'metadata_path':metadata_path,
    #                     'dataset':dataset,
    #                     'cword_config':cword_config,
    #                     'level':'dataset',
    #                     'verbose':True}
    #     data_object = generate_class(data)
    #     data_object.calculate_zscore()
        
    data = {}
    Input = []
    if Type=='registration':
        temp_input1 = []
        temp_input2 = []
    """ Process Dataset """
    if Type=='dataset':
        data = {'metadata_path':metadata_path,
                        'dataset':dataset,
                        'cword_config':cword_config,
                        'level':Type,
                        'verbose':True}
        Input.append(data)
    else:
        fishdata_path = os.path.join(metadata_path,'fishdata')
        if rerun:
            posnames = np.unique([i[len(dataset)+1:].split('_hybe')[0] for i in os.listdir(fishdata_path) if '.tif' in i])
            posnames = posnames[posnames!='X']    
            zindexes = np.unique([i.split('_')[-2] for i in os.listdir(fishdata_path) if '.tif' in i])
            zindexes = zindexes[zindexes!='cytoplasm']
            zindexes = zindexes[zindexes!='nuclei']
        else:
            nucstain = [i for i in os.listdir(metadata_path) if 'nucstain' in i][-1]
            image_metadata = Metadata(os.path.join(metadata_path,nucstain))
            posnames = image_metadata.image_table.Position.unique()
            zindexes = len(image_metadata.image_table.Zindex.unique())
            if config.parameters['projection_zstart'] == -1:
                config.parameters['projection_zstart'] = 0+config.parameters['projection_k']
            if config.parameters['projection_zend'] == -1:
                config.parameters['projection_zend'] = np.max(zindexes)-config.parameters['projection_k']
            zindexes = np.array(range(config.parameters['projection_zstart'],
                                      config.parameters['projection_zend'],
                                      config.parameters['projection_zskip']))
        for posname in posnames:
            if Type in ['position','segmentation']:
                data = {'metadata_path':metadata_path,
                            'dataset':dataset,
                            'posname':posname,
                            'cword_config':cword_config,
                            'level':Type,
                            'verbose':verbose}
                Input.append(data)
            elif Type=='classification':
                for zindex in zindexes:
                    data = {'metadata_path':metadata_path,
                                'dataset':dataset,
                                'posname':posname,
                                'zindex':zindex,
                                'cword_config':cword_config,
                                'level':Type,
                                'verbose':verbose}
                    Input.append(data)
            else:
                for seq,hybe,channel in bitmap:
                    if Type in ['hybe','registration','stack']:
                        if Type=='registration':
                            channel=config.parameters['registration_channel']
                        data = {'metadata_path':metadata_path,
                                'dataset':dataset,
                                'posname':posname,
                                'hybe':hybe,
                                'channel':channel,
                                'cword_config':cword_config,
                                'level':Type,
                                'verbose':verbose}
                        if Type=='registration':
                            if hybe==config.parameters['ref_hybe']:
                                temp_input1.append(data)
                            else:
                                temp_input2.append(data)
                        else:
                            Input.append(data)
                    elif Type=='image':
                        for zindex in zindexes:
                            data = {'metadata_path':metadata_path,
                                    'dataset':dataset,
                                    'posname':posname,
                                    'hybe':hybe,
                                    'channel':channel,
                                    'zindex':zindex,
                                    'cword_config':cword_config,
                                    'level':Type,
                                    'verbose':verbose}
                            Input.append(data)
    if Type=='registration':
        """ Required to find tforms for hybes that were processed before their references"""
        Input.extend(temp_input1)
        Input.extend(temp_input2)
    print(len(Input))
    if ncpu==1:
        iterable = tqdm(Input,total=len(Input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+dataset,position=0)
        for i in iterable:
            i = manual_class(i)
    else:
        temp_input = []
        for i in Input:
            temp_input.append(i)
            if len(temp_input)>=batches:
                pool = multiprocessing.Pool(ncpu)
                sys.stdout.flush()
                results = pool.imap(manual_class, temp_input)
                iterable = tqdm(results,total=len(temp_input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+dataset,position=0)
                for i in iterable:
                    pass
                pool.close()
                sys.stdout.flush()
                temp_input = []
        pool = multiprocessing.Pool(ncpu)
        sys.stdout.flush()
        results = pool.imap(manual_class, temp_input)
        iterable = tqdm(results,total=len(temp_input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+dataset,position=0)
        for i in iterable:
            pass
        pool.close()
        sys.stdout.flush()
        temp_input = []