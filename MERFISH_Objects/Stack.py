from tqdm import tqdm
from MERFISH_Objects.Image import *
from MERFISH_Objects.Deconvolution import *
from MERFISH_Objects.Utilities import *
import dill as pickle
import os
import numpy as np
import time
import importlib
from MERFISH_Objects.FISHData import *
from datetime import datetime

class Stack_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 hybe,
                 channel,
                 cword_config,
                 wait=300,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.channel = channel
        self.hybe = hybe
        self.wait = wait
        self.verbose = verbose
        
        self.acq = [i for i in os.listdir(self.metadata_path) if self.hybe in i][0]
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.psf_dict = self.merfish_config.psf_dict
        self.chromatic_dict = self.merfish_config.chromatic_dict
        self.parameters = self.merfish_config.parameters
        self.daemon_path = self.parameters['daemon_path']
        self.image_daemon_path = os.path.join(self.daemon_path,'image')
        if not os.path.exists(self.image_daemon_path):
            os.mkdir(self.image_daemon_path)
            os.mkdir(os.path.join(self.image_daemon_path,'input'))
            os.mkdir(os.path.join(self.image_daemon_path,'output'))
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        self.two_dimensional = self.parameters['two_dimensional']
        self.completed = False
        self.passed = True
        self.images_completed=False
        self.deconvolution_completed=False
        
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.utilities = Utilities_Class(self.parameters['utilities_path'])

    def run(self):
        self.check_flags()
    
    def update_user(self,message):
        """ For User Display"""
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
            
    def check_flags(self):
        if self.verbose:
            self.update_user('Checking Flags')
        self.failed = False
        #Position
        flag = self.utilities.load_data(Dataset=self.dataset,
                                        Position=self.posname,
                                        Type='flag')
        if flag == 'Failed':
            log = self.posname+' Failed'
            self.completed = True
            self.failed = True
        #Hybe
        if not self.failed:
            flag = self.utilities.load_data(Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Type='flag')
            if flag == 'Failed':
                log = self.hybe+' Failed'
                self.completed = True
                self.failed = True
        if self.failed:
            self.utilities.save_data('Failed',
                                    Dataset=self.dataset,
                                    Position=self.posname,
                                    Hybe=self.hybe,
                                    Channel=self.channel,
                                    Type='flag')
            self.utilities.save_data(log,
                                    Dataset=self.dataset,
                                    Position=self.posname,
                                    Hybe=self.hybe,
                                    Channel=self.channel,
                                    Type='log')
        #Stack
        if not self.failed:
            flag = self.utilities.load_data(Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Type='flag')
            if flag == 'Failed':
                log = self.channel+' Failed'
                self.completed = True
                self.failed = True
            elif flag == 'Passed':
                self.completed = True
                self.failed = False
            
        if not self.failed:
            self.check_image_flags()
                    
    def check_image_flags(self):
        self.images_completed = False
        self.check_projection()
        if self.verbose:
            iterable = tqdm(self.zindexes,desc='Checking Image Flags')
        else:
            iterable = self.zindexes
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for zindex in iterable:
            zindex = str(zindex)
            flag = self.utilities.load_data(Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Zindex=zindex,
                                            Type='flag')
            if isinstance(flag,type(None)):
                self.not_started.append(zindex)
            elif flag == 'Started':
                fname = self.dataset+'_'+self.posname+'_'+self.hybe+'_'+self.channel+'_'+str(zindex)+'.pkl'
                fname_path = os.path.join(self.image_daemon_path,'input',fname)
                if os.path.exists(fname_path):
                    self.started.append(zindex)
                else:
                    self.not_started.append(zindex)
            elif flag == 'Passed':
                self.passed.append(zindex)
            elif flag =='Failed':
                self.failed.append(zindex)
                """Not sure what to do here yet"""
        if len(self.not_started)==0: # All Images have been started
            if len(self.started)==0: # All Images have been completed
                self.images_completed = True
                self.completed = True
                self.utilities.save_data('Passed',
                                            Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Type='flag')
        else:
            self.create_images()
            
    def create_images(self):
        if not self.images_completed:
            if self.verbose:
                iterable = tqdm(self.not_started,desc='Creating Images')
            else:
                iterable = self.not_started
            for zindex in iterable:
                fname = self.dataset+'_'+self.posname+'_'+self.hybe+'_'+self.channel+'_'+str(zindex)+'.pkl'
                fname_path = os.path.join(self.image_daemon_path,'input',fname)
                data = {'metadata_path':self.metadata_path,
                        'dataset':self.dataset,
                        'posname':self.posname,
                        'hybe':self.hybe,
                        'channel':self.channel,
                        'zindex':str(zindex),
                        'cword_config':self.cword_config,
                        'level':'image'}
                pickle.dump(data,open(fname_path,'wb'))
                self.utilities.save_data('Started',
                                        Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Channel=self.channel,
                                        Zindex=str(zindex),
                                        Type='flag')
            
    def check_projection(self):
        if self.verbose:
            self.update_user('Checking Projection Zindexes')
#         self.metadata = Metadata(os.path.join(self.metadata_path,self.acq))
        self.image_table = pd.read_csv(os.path.join(self.metadata_path,self.acq,'Metadata.txt'),sep='\t')
        self.len_z = len(self.image_table[(self.image_table.Position==self.posname)].Zindex.unique())
        if self.projection_function=='None':
            self.projection_k = 0
        if self.projection_zstart==-1:
            self.projection_zstart = 0+self.projection_k
        elif self.projection_zstart>self.len_z:
            print('zstart of ',self.projection_zstart,' is larger than stk range of', self.len_z)
            raise(ValueError('Projection Error'))
        if self.projection_zend==-1:
            self.projection_zend = self.len_z-self.projection_k
        elif self.projection_zend>self.len_z:
            print('zend of ',self.projection_zend,' is larger than stk range of', self.len_z)
            raise(ValueError('Projection Error'))
        elif self.projection_zend<self.projection_zstart:
            print('zstart of ',self.projection_zstart,' is larger than zend of', self.projection_zend)
            raise(ValueError('Projection Error'))
        self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
        if self.two_dimensional:
            self.zindexes = [0]
