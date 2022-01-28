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
from datetime import datetime

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
        flag = self.utilities.load_data(Dataset=self.dataset,Position=self.posname,Type='flag')
        if flag == 'Failed':
            log = self.posname+' Failed'
            self.completed = True
            self.failed = True
        if self.failed:
            self.completed = True
            self.utilities.save_data('Failed',Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='flag')
            self.utilities.save_data(log,Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='log')
        #Hybe
        if not self.failed:
            flag = self.utilities.load_data(Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='flag')
            if flag == 'Failed':
                log = self.hybe+' Failed'
                self.completed = True
                self.failed = True
                
        if not self.failed:
            self.check_registration()
            
    def check_registration(self):
        if self.verbose:
            self.update_user('Checking Registration Flags')
        flag = self.utilities.load_data(Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Channel=self.parameters['registration_channel'],
                                        Type='flag')
        if isinstance(flag,type(None)):
            self.create_registration()
        elif flag == 'Started':
            fname = self.dataset+'_'+self.posname+'_'+self.hybe+'.pkl'
            fname_path = os.path.join(self.registration_daemon_path,'input',fname)
            if not os.path.exists(fname_path):
                self.create_registration()
            pass
        elif flag == 'Passed':
            self.check_stacks()
        elif flag =='Failed':
            self.utilities.save_data('Failed',Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='flag')
            self.utilities.save_data('Registration Failed',Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='log')
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
            flag = self.utilities.load_data(Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Channel=channel,Type='flag')
            if isinstance(flag,type(None)):
                self.not_started.append(channel)
            elif flag=='Started':
                fname = self.dataset+'_'+self.posname+'_'+self.hybe+'_'+channel+'.pkl'
                fname_path = os.path.join(self.stk_daemon_path,'input',fname)
                if os.path.exists(fname_path):
                    self.started.append(channel)
                else:
                    self.not_started.append(channel)
            elif flag=='Passed':
                self.passed.append(channel)
            elif flag =='Failed':
                self.failed.append(channel)
                self.utilities.save_data('Failed',Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='flag')
                self.utilities.save_data(str(channel+' Failed'),Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='log')
                self.completed = True
        if not self.completed:
            if len(self.not_started)==0: # All channels started
                if len(self.started)==0: # All channels finished
                    self.completed = True
                    self.utilities.save_data('Passed',Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='flag')
            else:
                self.create_stacks()
            
    def create_registration(self):
        if self.verbose:
            self.update_user('Checking Registration')
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
        self.utilities.save_data('Started',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Channel=self.parameters['registration_channel'],
                                 Type='flag')

            
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
            self.utilities.save_data('Started',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Channel=channel,
                                 Type='flag')
  