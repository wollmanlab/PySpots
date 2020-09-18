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
        self.ref_hybe = self.parameters['ref_hybe']
        self.registration_daemon_path = os.path.join(self.parameters['daemon_path'],'registration')
        self.stk_daemon_path = os.path.join(self.parameters['daemon_path'],'stack')
        self.utilities_path = self.parameters['utilities_path']
        
        self.completed = False
        self.passed = True

    def run(self):
        self.check_flags()
        
    def check_flags(self):
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        if flag=='Failed':
            self.completed = True
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname)
        elif flag=='Passed':
            self.completed = True
        else:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname)
            if flag=='Failed':
                self.completed = True
                self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
                self.fishdata.add_and_save_data(str(self.posname+' Failed'),'log',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
            elif flag=='Passed':
                self.completed = True
            else:
                self.check_registration()
            
    def check_registration(self):
        if self.verbose:
            for i in tqdm([0],desc='Checking Registration Flags'):
                pass
        flag =  self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.parameters['registration_channel'])
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

#     def check_completed(self):
#         #Look for all stacks having been created
#         self.completed_not_processed_idx = [z for z,i in enumerate(self.bitmap) if self.hybe in i]
#         self.utilities = Utilities_Class(self.utilities_path)
#         self.completed = True
#         if self.verbose:
#             iterable = tqdm(self.completed_not_processed_idx,desc='Checking Completed Stacks')
#         else:
#             iterable = self.completed_not_processed_idx
#         for bitmap_idx in iterable:
#             seq, hybe, channel = self.bitmap[bitmap_idx]
#             stk = self.utilities.load_data(Dataset=self.dataset,
#                                                Position=self.posname,
#                                                Hybe=self.hybe,
#                                                Channel=channel,
#                                                Type='stack')
#             if isinstance(stk,type(None)):
#                 self.completed = False
        
#     def feed_registration_daemon(self):
#         fname = str(self.dataset+'_'+self.posname+'_'+self.hybe+'.pkl')
#         if self.verbose:
#             iterable = tqdm([fname],desc = 'Generating Registration Class')
#         else:
#             iterable = [fname]
#         for fname in iterable:
#             reg_class = Registration_Class(self.metadata_path,
#                                            self.dataset,
#                                            self.posname,
#                                            self.hybe,
#                                            self.cword_config)
#             pickle.dump(reg_class,open(os.path.join(self.registration_daemon_path,'input',fname),'wb'))
#             self.reg_fnames = [fname]
#             self.registration_started = True
        
#     def check_registration_daemon(self):
#         fname = self.reg_fnames[0]
#         fname_path = os.path.join(self.registration_daemon_path,'output',fname)
#         if os.path.exists(fname_path):
#             if self.verbose:
#                 iterable = tqdm(self.reg_fnames,desc='Checking Registration Class')
#             else:
#                 iterable =  self.reg_fnames
#             for fname in iterable:
#                 fname_path = os.path.join(self.registration_daemon_path,'output',fname)
#                 wait = True
#                 start = time.time()
#                 while wait:
#                     try: # In case file is being written to
#                         reg_class = pickle.load(open(fname_path,'rb'))
#                         if not reg_class.passed:
#                             self.passed = False
#                         self.registration_completed = True
#                         wait = False
#                     except Exception as e:
#                         if (time.time()-start)>self.wait:
#                             wait = False
#                             if self.verbose:
#                                 print(fname_path)
#                                 print(e)
#                                 raise ValueError('Taking too long to load')
                
#     def feed_stk_daemon(self):
#         self.completed_not_processed_idx = [z for z,i in enumerate(self.bitmap) if self.hybe in i]
#         self.stk_fnames = [str(self.dataset+'_'+self.posname+'_'+hybe+'_'+channel+'.pkl') for seq, hybe, channel in self.bitmap if self.hybe==hybe]
#         not_completed = [fname for fname in self.stk_fnames if not os.path.exists(os.path.join(self.stk_daemon_path,'output',fname))]
#         not_completed = [fname for fname in not_completed if not os.path.exists(os.path.join(self.stk_daemon_path,'input',fname))]
#         if len(not_completed)>0:
#             if self.verbose:
#                 iterable = tqdm(self.completed_not_processed_idx,desc='Generating Stack Classes')
#             else:
#                 iterable = self.completed_not_processed_idx
#             self.stk_fnames = []
#             for bitmap_idx in iterable:
#                 seq, hybe, channel = self.bitmap[bitmap_idx]
#                 fname = str(self.dataset+'_'+self.posname+'_'+hybe+'_'+channel+'.pkl')
#                 stk_class = Stack_Class(self.metadata_path,
#                                         self.dataset,
#                                         self.posname,
#                                         channel,
#                                         hybe,
#                                         self.cword_config)
#                 pickle.dump(stk_class,open(os.path.join(self.stk_daemon_path,'input',fname),'wb'))
#                 self.stk_fnames.append(fname)
#             self.stack_started = True
            
#     def check_stk_daemon(self):
#         not_completed = [fname for fname in self.stk_fnames if not os.path.exists(os.path.join(self.stk_daemon_path,'output',fname))]
#         if len(not_completed) == 0:
#             if self.verbose:
#                 iterable = tqdm(self.stk_fnames,desc='Checking Stack Classes')
#             else:
#                 iterable = self.stk_fnames
#             for fname in iterable:
#                 fname_path = os.path.join(self.stk_daemon_path,'output',fname)
#                 start = time.time()
#                 wait = True
#                 while wait:
#                     try: # In case file is being written to
#                         stk_class = pickle.load(open(fname_path,'rb'))
#                         wait = False
#                         if not stk_class.passed:
#                             self.passed = False
#                     except Exception as e:
#                         if (time.time()-start)>self.wait:
#                             wait = False
#                             if self.verbose:
#                                 print(fname_path)
#                                 print(e)
#                                 raise ValueError('Timed Out')
                    
        