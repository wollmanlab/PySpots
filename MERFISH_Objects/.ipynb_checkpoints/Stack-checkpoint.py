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
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.psf_dict = self.merfish_config.psf_dict
        self.chromatic_dict = self.merfish_config.chromatic_dict
        self.parameters = self.merfish_config.parameters
        self.daemon_path = self.parameters['daemon_path']
        self.image_daemon_path = os.path.join(self.daemon_path,'image')
        self.decon_daemon_path = os.path.join(self.daemon_path,'deconvolution')
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        
        self.completed = False
        self.passed = True
        self.images_completed=False
        self.deconvolution_completed=False

    def run(self):
        self.check_flags()
            
    def check_flags(self):
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
        self.proceed = True
        if flag=='Failed': # Self Failed
            self.completed = True
        elif flag=='Passed':
            self.completed = True
        else:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
            if flag=='Failed': # Hybe Failed
                self.completed = True
                self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                self.fishdata.add_and_save_data(str(self.hybe)+' Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
            elif flag=='Passed':
                self.completed = True
            else:
                flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname)
                if flag=='Failed': # Position Failed
                    self.completed = True
                    self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                    self.fishdata.add_and_save_data(str(self.hybe)+' Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                    self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
                    self.fishdata.add_and_save_data(str(self.posname)+' Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
                elif flag=='Passed':
                    self.completed = True
                else:
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
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=zindex)
            if isinstance(flag,type(None)):
                self.not_started.append(zindex)
            elif flag == 'Started':
                self.started.append(zindex)
            elif flag == 'Passed':
                self.passed.append(zindex)
            elif flag =='Failed':
                self.failed.append(zindex)
                # Not sure what to do here yet ********
        if len(self.not_started)==0: # All Images have been started
            if len(self.started)==0: # All Images have been completed
                self.images_completed = True
                self.check_deconvolution_flags()
        else:
            self.check_images()
                
    def check_deconvolution_flags(self):
        if self.verbose:
            print('Checking Deconvolution Flags')
        flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex='all')
        if flag =='Started':
            pass
        elif flag=='Passed':
            self.completed = True
            self.fishdata.add_and_save_data('Passed',
                                            'flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe=self.hybe,
                                            channel=self.channel,
                                            zindex='all')
        elif flag=='Failed':
            self.deconvolution_completed = True
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
            self.fishdata.add_and_save_data('Deconvolution Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
            self.fishdata.add_and_save_data(str(self.channel)+' Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname)
            self.fishdata.add_and_save_data(str(self.hybe)+' Failed','log',dataset=self.dataset,posname=self.posname)
            self.completed = True
        else:
            self.check_deconvolution()
            
    def check_images(self):
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
                self.flag = self.fishdata.add_and_save_data('Started','flag',
                                                            dataset=self.dataset,
                                                            posname=self.posname,
                                                            hybe=self.hybe,
                                                            channel=self.channel,
                                                            zindex=str(zindex))
    def check_deconvolution(self):
        fname = self.dataset+'_'+self.posname+'_'+self.hybe+'_'+self.channel+'.pkl'
        fname_path = os.path.join(self.decon_daemon_path,'input',fname)
        data = {'metadata_path':self.metadata_path,
                'dataset':self.dataset,
                'posname':self.posname,
                'hybe':self.hybe,
                'channel':self.channel,
                'cword_config':self.cword_config,
                'level':'deconvolution'}
        pickle.dump(data,open(fname_path,'wb'))
        self.fishdata.add_and_save_data('Started','flag',
                                                    dataset=self.dataset,
                                                    posname=self.posname,
                                                    hybe=self.hybe,
                                                    channel=self.channel,
                                                    zindex='all')
        
#     def load_tforms(self):
#         self.hybedata_path = os.path.join(self.metadata_path,self.parameters['hybedata'],self.posname)
#         self.hybedata = HybeData(self.hybedata_path)
#         self.translation = self.hybedata.load_data(self.posname, self.hybe, 'tforms')
#         self.translation_z = self.translation['z']
            
    def check_projection(self):
        if self.verbose:
            print('Checking Projection Zindexes')
        self.metadata = Metadata(self.metadata_path)
        self.len_z = len(self.metadata.image_table[(self.metadata.image_table.Position==self.posname)].Zindex.unique())
        del self.metadata
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
        elif zend<zstart:
            print('zstart of ',self.projection_zstart,' is larger than zend of', self.projection_zend)
            raise(ValueError('Projection Error'))
        self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
            
#     def feed_image_daemon(self):
#         self.check_projection()
#         self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
#         self.len_z = len(self.zindexes)
#         if self.verbose:
#             iterable = tqdm(self.zindexes,total=len(self.zindexes),desc='Generating Image Classes')
#         else:
#             iterable = self.zindexes
#         self.sub_stk_fnames = []
#         for z in iterable:
#             zindexes = list(np.array(range(z-self.projection_k,z+self.projection_k+1))+round(self.translation_z))
#             fname = str(self.dataset+'_'+self.posname+'_'+self.hybe+'_'+self.channel+'_'+str(z)+'.pkl')
#             img_class = Image_Class(self.metadata_path,
#                                     self.dataset,
#                                     self.posname,
#                                     z,
#                                     zindexes,
#                                     self.channel,
#                                     self.hybe,
#                                     self.cword_config)
#             self.sub_stk_fnames.append(fname)
#             pickle.dump(img_class,open(os.path.join(self.image_daemon_path,'input',fname),'wb'))
#         self.image_started = True
            
#     def check_image_daemon(self):
#         not_completed = [fname for fname in self.sub_stk_fnames if not os.path.exists(os.path.join(self.image_daemon_path,'output',fname))]
#         if len(not_completed)==0:
#             if self.verbose:
#                 iterable = tqdm(self.sub_stk_fnames,desc='Loading Image Classes')
#             else:
#                 iterable = self.sub_stk_fnames
#             for fname in iterable:
#                 fname_path = os.path.join(self.image_daemon_path,'output',fname)
#                 wait = True
#                 start = time.time()
#                 while wait:
#                     try: # In case file is being written to
#                         img_class = pickle.load(open(fname_path,'rb'))
#                         wait = False
#                         if not img_class.passed:
#                             self.passed = False
#                     except Exception as e:
#                         if (time.time()-start)>self.wait:
#                             wait = False
#                             if self.verbose:
#                                 print(fname_path)
#                                 print(e)
#                                 raise 'Timed out'
#             self.image_completed=True

#     def feed_decon_daemon(self):
#         if self.verbose:
#             for i in tqdm([1],desc='Creating Deconvolution Class'):
#                 pass
#         fname = str(self.dataset+'_'+self.posname+'_'+self.hybe+'_'+self.channel+'.pkl')
#         decon_class = Deconvolution_Class(self.metadata_path,
#                                           self.dataset,
#                                           self.posname,
#                                           self.channel,
#                                           self.hybe,
#                                           self.zindexes,
#                                           self.cword_config)
#         pickle.dump(decon_class,open(os.path.join(self.decon_daemon_path,'input',fname),'wb'))
#         self.decon_fnames = [fname]
#         self.deconvolution_started = True
        
#     def check_decon_daemon(self):
#         if self.verbose:
#             iterable = tqdm(self.decon_fnames,desc='Checking Deconvolution Class')
#         else:
#             iterable = self.decon_fnames
#         for fname in self.decon_fnames:
#             fname_path = os.path.join(self.decon_daemon_path,'output',fname)
#             if os.path.exists(fname_path):
#                 try: # In case file is being written to
#                     decon_class = pickle.load(open(fname_path,'rb'))
#                     if decon_class.completed:
#                         self.deconvolution_completed = True
#                         self.completed = True
#                         if not decon_class.passed:
#                             self.passed = False
#                 except Exception as e:
#                     if self.verbose:
#                         print(fname_path)
#                         print(e)
