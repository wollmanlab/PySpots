from tqdm import tqdm
from metadata import Metadata
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Image import *
from MERFISH_Objects.Deconvolution import *
from hybescope_config.microscope_config import *
from MERFISH_Objects.FISHData import *
import dill as pickle
import os
import importlib

class Hybe_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 hybe,
                 cword_config,
                 wait=300,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.hybe = hybe
        self.verbose = verbose
        self.wait = wait
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.bitmap = self.merfish_config.bitmap
        self.parameters = self.merfish_config.parameters
        self.registration_channel = self.parameters['registration_channel']
        self.registration_daemon_path = os.path.join(self.parameters['daemon_path'],'registration')
        if not os.path.exists(self.registration_daemon_path):
            os.mkdir(self.registration_daemon_path)
            os.mkdir(os.path.join(self.registration_daemon_path,'input'))
            os.mkdir(os.path.join(self.registration_daemon_path,'output'))
        self.stk_daemon_path = os.path.join(self.parameters['daemon_path'],'stack')
        if not os.path.exists(self.stk_daemon_path):
            os.mkdir(self.stk_daemon_path)
            os.mkdir(os.path.join(self.stk_daemon_path,'input'))
            os.mkdir(os.path.join(self.stk_daemon_path,'output'))
        
        self.completed = False
        self.passed = True

    def run(self):
        self.check_flags()
        
    def check_flags(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Flags')]
        self.failed = False
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        #Position
        flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                       posname=self.posname)
        if flag == 'Failed':
            log = self.posname+' Failed'
            self.completed = True
            self.failed = True
        if self.failed:
            self.completed = True
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                posname=self.posname,hybe=self.hybe)
            self.fishdata.add_and_save_data(log,'log',
                                                dataset=self.dataset,posname=self.posname,
                                                hybe=self.hybe)
        #Hybe
        if not self.failed:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                           posname=self.posname,hybe=self.hybe)
            if flag == 'Failed':
                log = self.hybe+' Failed'
                self.completed = True
                self.failed = True
                
        if not self.failed:
            self.check_registration()
            
    def check_registration(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Registration Flags')]
        flag =  self.fishdata.load_data('flag',dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.parameters['registration_channel'])
        if isinstance(flag,type(None)):
            self.create_registration()
        elif flag == 'Started':
            pass
        elif flag == 'Passed':
            self.check_stacks()
        elif flag =='Failed':
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
            self.fishdata.add_and_save_data('Registration Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
            self.completed = True 

    def check_stacks(self):
        if self.hybe == 'nucstain':
            self.channels = np.unique([channel for seq,hybe,channel in self.bitmap])
        else:
            self.channels = [channel for seq,hybe,channel in self.bitmap if self.hybe==hybe]
        if self.verbose:
            iterable = tqdm(self.channels,desc='Checking Stack Flags')
        else:
            iterable = self.channels
        self.not_started = []
        self.started = []
        self.passed = []
        self.failed = []
        for channel in iterable:
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=channel)
            if isinstance(flag,type(None)):
                self.not_started.append(channel)
            elif flag=='Started':
                self.started.append(channel)
            elif flag=='Passed':
                self.passed.append(channel)
            elif flag =='Failed':
                self.failed.append(channel)
                self.fishdata.add_and_save_data('Failed','flag',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                hybe=self.hybe)
                self.fishdata.add_and_save_data(str(channel+' Failed'),'log',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                hybe=self.hybe)
                self.completed = True
        if not self.completed:
            if len(self.not_started)==0: # All channels started
                if len(self.started)==0: # All channels finished
                    self.completed = True
                    self.fishdata.add_and_save_data('Passed','flag',
                                                    dataset=self.dataset,
                                                    posname=self.posname,
                                                    hybe=self.hybe)
            else:
                self.create_stacks()
            
    def create_registration(self):
        if self.verbose:
            for i in tqdm([0],desc='Creating Registration'):
                pass
        fname = self.dataset+'_'+self.posname+'_'+self.hybe+'.pkl'
        fname_path = os.path.join(self.registration_daemon_path,'input',fname)
        data = {'metadata_path':self.metadata_path,
                'dataset':self.dataset,
                'posname':self.posname,
                'hybe':self.hybe,
                'channel':self.parameters['registration_channel'],
                'cword_config':self.cword_config,
                'level':'registration'}
        pickle.dump(data,open(fname_path,'wb'))
        self.flag = self.fishdata.add_and_save_data('Started','flag',
                                                    dataset=self.dataset,
                                                    posname=self.posname,
                                                    hybe=self.hybe,
                                                    channel=self.parameters['registration_channel'])
            
    def create_stacks(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc='Creating Stacks')
        else:
            iterable = self.not_started
        for channel in iterable:
            fname = self.dataset+'_'+self.posname+'_'+self.hybe+'_'+channel+'.pkl'
            fname_path = os.path.join(self.stk_daemon_path,'input',fname)
            data = {'metadata_path':self.metadata_path,
                'dataset':self.dataset,
                'posname':self.posname,
                'hybe':self.hybe,
                'channel':channel,
                'cword_config':self.cword_config,
                'level':'stack'}
            pickle.dump(data,open(fname_path,'wb'))
            self.flag = self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname,
                                                        hybe=self.hybe,
                                                        channel=channel)
  