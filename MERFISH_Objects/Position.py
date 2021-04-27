from MERFISH_Objects.Hybe import *
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Image import *
from MERFISH_Objects.Deconvolution import *
from MERFISH_Objects.FISHData import *
from tqdm import tqdm
import dill as pickle
import time
import os
import numpy as np
from metadata import Metadata
from fish_results import HybeData

class Position_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 fresh=True,
                 wait=300,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.fresh = fresh
        self.verbose = verbose
        self.wait = wait
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.utilities_path = self.parameters['utilities_path']
        self.normalization_max = self.parameters['normalization_max']
        self.daemon_path = self.parameters['daemon_path']
        self.hybe_daemon_path = os.path.join(self.daemon_path,'hybe')
        self.ref_hybe = self.parameters['ref_hybe']
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        self.hybedata_path = os.path.join(self.metadata_path,self.parameters['hybedata'],self.posname)
        self.nbits = self.merfish_config.nbits
        self.bitmap = self.merfish_config.bitmap
        self.len_x = 2048
        self.len_y = 2048
        self.all_hybes = list(np.unique([hybe for seq,hybe,channel in self.bitmap]))
        self.hybe_fnames_list = [str(self.dataset+'_'+self.posname+'_'+hybe+'.pkl') for hybe in self.all_hybes]
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        
        self.segmentation_completed = False
        self.hybes_completed = False
        self.classification_completed = False
        self.completed = False
        self.passed = True
        self.failed = False
        
    def run(self):
        self.check_flag()

    def check_flag(self):
        if self.verbose:
            tqdm([],desc='Checking Flag')
        flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname)
        if flag == 'Failed':
            self.failed = True
            self.completed = True
        if not self.failed:
            self.check_imaging()
                    
    def check_imaging(self):
        if self.verbose:
            tqdm([],desc='Checking Imaging')
        self.started_acqs = [i for i in os.listdir(self.metadata_path) if 'hybe' in i]
        self.nucstain = [i for i in os.listdir(self.metadata_path) if 'nucstain' in i]
        dictionary = {''.join([i+'_' for i in os.listdir(os.path.join(self.metadata_path,self.started_acqs[0],posname_directory))[0][4:].split('_')[:-5]])[:-1]:posname_directory for posname_directory in os.listdir(os.path.join(self.metadata_path,self.started_acqs[0])) if not 'Metadata' in posname_directory}
        
        self.acqs = [i for i in self.started_acqs if os.path.exists(os.path.join(self.metadata_path,i,dictionary[self.posname]))]
        self.imaged_hybes = [i.split('_')[0] for i in self.acqs if 'hybe' in i]
        self.not_imaged = list(np.unique([hybe for seq,hybe,channel in self.bitmap if not hybe in self.imaged_hybes]))
        if len(self.nucstain)>0:
            self.check_segmentation()
#             self.imaged_hybes.append('nucstain')
        if len(self.imaged_hybes)>0:
            self.check_hybes()
        if self.hybes_completed:
            if self.segmentation_completed:
                self.check_classification()
        if self.classification_completed:
            self.completed = True
            self.fishdata.add_and_save_data('Passed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe='all')

    def check_segmentation(self):
        """ check segmentation flag"""
        flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                       posname=self.posname,
                                       channel=self.parameters['nucstain_channel'])
        if self.verbose:
                tqdm([],desc='Checking Segmentation')
        if flag == 'Started':
            do = 'nothing'
        elif flag =='Failed':
            self.fishdata.add_and_save_data('Failed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname)
            self.fishdata.add_and_save_data('Segmentation Failed','log',
                                            dataset=self.dataset,
                                            posname=self.posname)
        elif flag =='Passed':
            self.segmentation_completed = True
        else:
            self.create_segmentation()
        
    def create_segmentation(self):
        """create segmentation object"""
        self.segmentation_daemon_path = os.path.join(self.daemon_path,'segmentation')
        if not os.path.exists(self.segmentation_daemon_path):
                os.mkdir(self.segmentation_daemon_path)
                os.mkdir(os.path.join(self.segmentation_daemon_path,'input'))
                os.mkdir(os.path.join(self.segmentation_daemon_path,'output'))
        fname = self.dataset+'_'+self.posname+'.pkl'
        fname_path = os.path.join(self.segmentation_daemon_path,'input',fname)
        data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':self.posname,
                    'cword_config':self.cword_config,
                    'level':'segmentation'}
        pickle.dump(data,open(fname_path,'wb'))
        self.fishdata.add_and_save_data('Started','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            channel=self.parameters['nucstain_channel'])
        
    def check_hybes(self):
        if self.verbose:
            iterable = tqdm(self.imaged_hybes,desc='Checking Hybe Flags')
        else:
            iterable = self.imaged_hybes
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for hybe in iterable:
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,
                                            posname=self.posname,hybe=hybe)
            if isinstance(flag,type(None)):
                self.not_started.append(hybe)
            elif flag == 'Started':
                self.started.append(hybe)
            elif flag == 'Passed':
                self.passed.append(hybe)
            elif flag =='Failed':
                self.failed.append(hybe)
                self.fishdata.add_and_save_data('Failed','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
                self.fishdata.add_and_save_data(str(hybe+' Failed'),'log',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
                self.completed = True
        if not self.completed:
            if len(self.not_started)==0: # All hybes have been started
                if len(self.started)==0: # All imaged hybes have been completed
                    if len(self.not_imaged)==0:# All hybes have been imaged
                        if len(self.failed)==0: # No hybes failed
                            self.hybes_completed = True
            else:
                self.create_hybes()
                        
    def create_hybes(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc='Creating Hybes')
        else:
            iterable = self.not_started
        self.hybe_daemon_path = os.path.join(self.daemon_path,'hybe')
        if not os.path.exists(self.hybe_daemon_path):
                os.mkdir(self.hybe_daemon_path)
                os.mkdir(os.path.join(self.hybe_daemon_path,'input'))
                os.mkdir(os.path.join(self.hybe_daemon_path,'output'))
        for hybe in iterable:
            fname = self.dataset+'_'+self.posname+'_'+hybe+'.pkl'
            fname_path = os.path.join(self.hybe_daemon_path,'input',fname)
            data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':self.posname,
                    'hybe':hybe,
                    'cword_config':self.cword_config,
                    'level':'hybe'}
            pickle.dump(data,open(fname_path,'wb'))
            self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname,
                                                        hybe=hybe)
    def check_classification(self):
        if self.verbose:
            tqdm([],desc='Checking Classification')
        flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                       posname=self.posname,hybe='all')
        if isinstance(flag,type(None)):
            self.create_classification()
        if flag=='Passed':
            self.completed = True
            self.fishdata.add_and_save_data('Passed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname)
#         flag = self.fishdata.load_data('flag',dataset=self.dataset,hybe='all')
#         if flag == 'Passed':
#             flag = self.fishdata.load_data('flag',dataset=self.dataset,
#                                        posname=self.posname,hybe='all')
#             if flag == 'Started':
#                 do = 'nothing'
#             elif flag =='Failed':
#                 self.fishdata.add_and_save_data('Failed','flag',
#                                                 dataset=self.dataset,
#                                                 posname=self.posname)
#                 self.fishdata.add_and_save_data('Classification Failed','log',
#                                                 dataset=self.dataset,
#                                                 posname=self.posname)
#             elif flag =='Passed':
#                 self.classification_completed = True
#             else:
#                 self.create_classification()
        
    def create_classification(self):
        self.classification_daemon_path = os.path.join(self.daemon_path,'classification')
        if not os.path.exists(self.classification_daemon_path):
            os.mkdir(self.classification_daemon_path)
            os.mkdir(os.path.join(self.classification_daemon_path,'input'))
            os.mkdir(os.path.join(self.classification_daemon_path,'output'))
        fname = self.dataset+'_'+self.posname+'.pkl'
        fname_path = os.path.join(self.classification_daemon_path,'input',fname)
        data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':self.posname,
                    'cword_config':self.cword_config,
                    'level':'classification'}
        pickle.dump(data,open(fname_path,'wb'))
        self.fishdata.add_and_save_data('Started','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe='all')
        
        