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
        
#         self.imaging_completed = False
#         self.hybes_completed = False
#         self.saved = False
        self.completed = False
        self.passed = True
        
    def run(self):
        self.check_imaging()
        self.check_flags()
        self.create_hybes()

    def check_imaging(self):
        if self.verbose:
            for i in tqdm([1],desc='Checking Imaging'):
                pass
        self.metadata = Metadata(self.metadata_path)
        self.image_table = self.metadata.image_table[self.metadata.image_table.Position==self.posname].copy()
        self.image_zindexes = self.image_table[(self.image_table.Position==self.posname)].Zindex.unique()
        self.max_z = np.max(self.image_zindexes)
        self.acqs = list(self.image_table.acq.unique())
        self.imaged_hybes = [i.split('_')[0] for i in self.acqs if 'hybe' in i]
        self.nucstain = [i for i in self.acqs if 'nucstain' in i]
        self.not_imaged = list(np.unique([hybe for seq,hybe,channel in self.bitmap if not hybe in self.imaged_hybes]))
          
    def check_flags(self):
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata'],self.posname))
        if self.verbose:
            iterable = tqdm(self.imaged_hybes,desc='Checking Hybe Flags')
        else:
            iterable = self.imaged_hybes
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for hybe in iterable:
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=hybe)
            if isinstance(flag,type(None)):
                self.not_started.append(hybe)
            elif flag == 'Started':
                self.started.append(hybe)
            elif flag == 'Passed':
                self.passed.append(hybe)
            elif flag =='Failed':
                self.failed.append(hybe)
                self.flag = self.fishdata.add_and_save_data('Failed','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
                self.completed = True
        if not self.completed:
            if len(self.not_imaged)==0:# All hybes have been imaged
                if len(self.not_started)==0: # All hybes have been started
                    if len(self.started)==0: # All hybes have been completed
                        self.completed = True
                        if len(self.failed)==0:
                            self.flag = self.fishdata.add_and_save_data('Passed','flag',
                                                                dataset=self.dataset,
                                                                posname=self.posname)
    def create_hybes(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc='Creating Hybes')
        else:
            iterable = self.not_started
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
            self.flag = self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname,
                                                        hybe=hybe)
            
        
#     def check_completed(self):
#         self.utilities = Utilities_Class(self.utilities_path)
#         if self.verbose:
#             iterable = tqdm(self.bitmap,desc='Checking Completed Stacks')
#         else:
#             iterable = self.bitmap
#         self.not_completed_hybes = []
#         for seq, hybe, channel in iterable:
#             stk = self.utilities.load_data(Dataset=self.dataset,
#                                                Position=self.posname,
#                                                Hybe=hybe,
#                                                Channel=channel,
#                                                Type='stack')
#             if isinstance(stk,type(None)):
#                 self.not_completed_hybes.append(hybe)
#         self.not_completed_hybes = list(np.unique(self.not_completed_hybes))
#         if len(self.not_completed_hybes)>0:
#             self.hybes_completed = False
#         else:
#             self.hybes_completed = True
            
#     def check_imaging(self):
#         if self.verbose:
#             for i in tqdm([1],desc='Checking Imaging'):
#                 pass
#         self.metadata = Metadata(self.metadata_path)
#         self.image_table = self.metadata.image_table[self.metadata.image_table.Position==self.posname].copy()
#         self.image_zindexes = self.image_table[(self.image_table.Position==self.posname)].Zindex.unique()
#         del self.metadata
#         self.max_z = np.max(self.image_zindexes)
#         self.acqs = list(self.image_table.acq.unique())
#         self.imaged_hybes = [i.split('_')[0] for i in self.acqs if 'hybe' in i]
#         self.nucstain = [i for i in self.acqs if 'nucstain' in i]
#         self.not_imaged = list(np.unique([hybe for seq,hybe,channel in self.bitmap if not hybe in self.imaged_hybes]))
#         if len(self.not_imaged)==0:
#             self.imaging_completed = True
        
#     def check_projection(self):        
#         if self.projection_function=='None':
#             self.projection_k = 0
#         if self.projection_zstart==-1:
#             self.projection_zstart = 0+self.projection_k
#         elif self.projection_zstart>self.max_z:
#             print('zstart of ',self.projection_zstart,' is larger than stk range of', self.max_z)
#             raise(ValueError('Projection Error'))
#         if self.projection_zend==-1:
#             self.projection_zend = self.max_z-self.projection_k
#         elif self.projection_zend>self.max_z:
#             print('zend of ',self.projection_zend,' is larger than stk range of', self.max_z)
#             raise(ValueError('Projection Error'))
#         elif self.projection_zend<self.projection_zstart:
#             print('zstart of ',self.projection_zstart,' is larger than zend of', self.projection_zend)
#             raise(ValueError('Projection Error'))
#         self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
        
#     def feed_hybe_daemon(self):
#         not_completed = [fname for fname in self.hybe_fnames_list if not os.path.exists(os.path.join(self.hybe_daemon_path,'input',fname))]
#         not_completed = [fname for fname in not_completed if not os.path.exists(os.path.join(self.hybe_daemon_path,'output',fname))]
#         if len(not_completed)>0:
#             if self.verbose:
#                 iterable = tqdm(self.all_hybes,desc='Generating Hybe Classes')
#             else:
#                 iterable = self.all_hybes
#             for hybe in iterable:
#                 if not hybe in self.imaged_hybes:
#                     continue
#                 fname = str(self.dataset+'_'+self.posname+'_'+hybe+'.pkl')
#                 if os.path.exists(os.path.join(self.hybe_daemon_path,'input',fname)):
#                     continue
#                 elif os.path.exists(os.path.join(self.hybe_daemon_path,'output',fname)):
#                     continue
#                 else:
#                     hyb_class = Hybe_Class(self.metadata_path,
#                                            self.dataset,
#                                            self.posname,
#                                            hybe,
#                                            self.cword_config)
#                     pickle.dump(hyb_class,open(os.path.join(self.hybe_daemon_path,'input',fname),'wb'))
            
#     def check_hybe_daemon(self):
#         not_completed = [fname for fname in self.hybe_fnames_list if not os.path.exists(os.path.join(self.hybe_daemon_path,'output',fname))]
#         if len(not_completed)==0:
#             self.hybe_completed = True
#             if self.verbose:
#                 iterable = tqdm(self.hybe_fnames_list,desc='Checking Hybe Classes')
#             else:
#                 iterable = self.hybe_fnames_list
#             # All hybes are done check if they passed
#             for fname in iterable:
#                 fname_path = os.path.join(self.hybe_daemon_path,'output',fname)
#                 start = time.time()
#                 wait = True
#                 while wait: # In case File is bein written to
#                     try:
#                         hyb_class = pickle.load(open(fname_path,'rb'))
#                         if hyb_class.completed:
#                             if not hyb_class.passed:
#                                 self.passed = False
#                             wait = False
#                     except Exception as e:
#                         if (time.time()-start)>self.wait:
#                             wait = False
#                             self.passed = False
#                         if self.verbose:
#                             print(e)
#                         pass 
        
                    
#     def load_tesseract(self):
#         self.hybedata = HybeData(self.hybedata_path)
#         self.check_projection()
#         self.len_z = len(self.zindexes)
#         tesseract = np.zeros((self.len_x,self.len_y,self.len_z,self.nbits)).astype('uint16')
#         if not self.fresh:
#             zindexes = self.hybedata.metadata[self.hybedata.metadata.dtype=='cstk'].zindex.unique()
#             if self.verbose:
#                 iterable = tqdm(zindexes,desc='Loading Tesseract')
#             else:
#                 iterable = zindexes
#             for z in iterable:
#                 try:
#                     zindex = np.where(self.zindex==z)[0]
#                     tesseract[:,:,zindex,:] = self.hybedata.load_data(self.posname, z, 'cstk')
#                 except Exception as e:
#                     print(e)
#                     print(z)
#                     print(self.hybedata.load_data(self.posname, z, 'cstk'))
#         self.tesseract = tesseract
    
#     def populate_tesseract(self):
#         self.utilities = Utilities_Class(self.utilities_path)
#         if self.verbose:
#             iterable = tqdm(range(len(self.bitmap)),desc='Populating Tesseract')
#         else:
#             iterable = range(len(self.bitmap))
#         for bitmap_idx in iterable:
#             seq,hybe,channel = self.bitmap[bitmap_idx]
#             stk = self.utilities.load_data(Dataset=self.dataset,
#                                            Position=self.posname,
#                                            Hybe=hybe,
#                                            Channel=channel,
#                                            Type='stack')
#             if not isinstance(stk,type(None)):
#                 self.tesseract[:,:,:,bitmap_idx] = stk
#             else:
#                 print(self.metadata_path,self.dataset,self.posname,hybe,channel)
#                 raise(ValueError('some stacks arent done but they are marked as done'))

#     def save_tesseract(self):
# #         block = range(100)
# #         compleded_zindexes = [i for i in range(self.tesseract.shape[2]) if np.sum(self.tesseract[block,block,i,:].ravel())>0]
#         if self.verbose:
#             iterable = tqdm(range(self.tesseract.shape[2]),desc='Saving Tesseract')
#         else:
#             iterable = range(self.tesseract.shape[2])
#         for zindex in iterable:
#             z = self.zindexes[zindex]
#             self.hybedata.add_and_save_data(self.tesseract[:,:,zindex,:], self.posname, z, 'cstk')
#             self.hybedata.add_and_save_data(self.normalization_max*np.ones(self.nbits), self.posname, z, 'nf')
#             self.hybedata.add_and_save_data(np.zeros((self.len_x,self.len_y)), self.posname, z, 'cimg')
#         self.completed = True
#         self.saved = True
        
        