from tqdm import tqdm
from MERFISH_Objects.Position import *
import pandas as pd
from metadata import Metadata
from hybescope_config.microscope_config import *
import numpy as np
import dill as pickle
import importlib
from tqdm import tqdm
import cv2
import random
from skimage.filters import threshold_otsu
from MERFISH_Objects.FISHData import *

class Dataset_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 cword_config,
                 wait = 300,
                 verbose=True):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.cword_config = cword_config
        self.verbose = verbose
        self.wait = wait
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.daemon_path = self.parameters['daemon_path']
        self.position_daemon_path = os.path.join(self.daemon_path,'position')
        self.bitmap = self.merfish_config.bitmap
        self.channels = list(np.unique([channel for seq,hybe,channel in self.bitmap]))
        self.hybes = list(np.unique([hybe for seq,hybe,channel in self.bitmap]))
#         self.find_hot_pixels()
        
        self.completed = False
        self.passed = True
        
        
    def run(self):
        self.check_imaging()
        
    def check_imaging(self):
        self.metadata = Metadata(self.metadata_path)
        self.acqs = [i for i in self.metadata.image_table.acq.unique() if 'hybe' in i]
        self.posnames = self.metadata.image_table[self.metadata.image_table.acq.isin(self.acqs)].Position.unique()
        self.check_flags()
        
    def check_flags(self):
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.flag = self.fishdata.add_and_save_data('Started','flag',dataset=self.dataset)
        if self.verbose:
            iterable = tqdm(self.posnames,desc='Checking Position Flags')
        else:
            iterable = self.posnames
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for posname in iterable:
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,posname=posname)
            if isinstance(flag,type(None)):
                self.not_started.append(posname)
            elif flag == 'Started':
                self.started.append(posname)
            elif flag == 'Passed':
                self.passed.append(posname)
            elif flag =='Failed':
                self.failed.append(posname)
        if len(self.acqs)>1: # All positions have been imaged atleast once
            if len(self.not_started)==0: # All positions have been started
                if len(self.started)==0: # All positions have been completed
                    self.completed = True 
                    self.flag = self.fishdata.add_and_save_data('Passed','flag',
                                                        dataset=self.dataset)
            else:
                self.create_positions()
                
    def create_positions(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc='Creating Positions')
        else:
            iterable = self.not_started
        for posname in iterable:
            fname = self.dataset+'_'+posname+'.pkl'
            fname_path = os.path.join(self.position_daemon_path,'input',fname)
            data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':posname,
                    'cword_config':self.cword_config,
                    'level':'position'}
            pickle.dump(data,open(fname_path,'wb'))
            self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=posname)
            
    
#     def feed_pos_daemon(self):
#         not_completed = [fname for fname in self.pos_fnames if not os.path.exists(os.path.join(self.position_daemon_path,'input',fname))]
#         not_completed = [fname for fname in not_completed if not os.path.exists(os.path.join(self.position_daemon_path,'output',fname))]
#         if len(not_completed)>0:
#             if self.verbose:
#                 iterable = tqdm(self.posnames,desc='Generating Position Classes')
#             else:
#                 iterable = self.posnames
#             for posname in iterable:
#                 fname = self.dataset+'_'+posname+'.pkl'
#                 if os.path.exists(os.path.join(self.position_daemon_path,'input',fname)):
#                     continue
#                 elif os.path.exists(os.path.join(self.position_daemon_path,'output',fname)):
#                     continue
#                 else:
#                     pos_class = Position_Class(self.metadata_path,
#                                                self.dataset,
#                                                posname,
#                                                self.cword_config)
#                     pickle.dump(pos_class,open(os.path.join(self.position_daemon_path,'input',fname),'wb'))
                
#     def check_pos_daemon(self):
#         not_completed = [fname for fname in self.pos_fnames if not os.path.exists(os.path.join(self.position_daemon_path,'output',fname))]
#         if len(not_completed)==0:
#             # All positions are completed
#             if self.verbose:
#                 iterable = tqdm(self.pos_fnames,desc='Checking Position Classes')
#             else:
#                 iterable = self.pos_fnames
#             for fname in iterable:
#                 fname_path = os.path.join(self.position_daemon_path,'output',fname)
#                 start = time.time()
#                 wait = True
#                 while wait: # In case File is bein written to
#                     try:
#                         pos_class = pickle.load(open(fname_path,'rb'))
#                         if pos_class.passed:
#                             if pos_class.completed:
#                                 wait = False
#                                 # os.remove(fname_path)
#                             else:
#                                 print(fname_path)
#                                 raise(ValueError('Position is in output but not completed'))
#                         else:
#                             # What to do with failed positions
#                             wait = False
#                             # os.remove(fname_path)
#                     except Exception as e:
#                         if (time.time()-start)>self.wait:
#                             wait = False
#                             self.passed = False
#                         if self.verbose:
#                             print(fname_path)
#                             print(e)
#                         pass
#             self.completed = True
                
            
    def find_hot_pixels(self,n_pos=5,std_thresh=3,n_acqs=5,kernel_size=3):
        self.load_metadata()
        if kernel_size%2==0:
            kernel_size = kernel_size+1
        kernel = np.ones((kernel_size,kernel_size))
        kernel[int(kernel_size/2),int(kernel_size/2)] = 0
        kernel = kernel/np.sum(kernel)
        X = []
        Y = []
        hot_pixel_dict = {}
        if len(self.poses)>n_pos:
            pos_sample = random.sample(list(self.poses),n_pos)
        else:
            pos_sample = self.poses
        if self.verbose:
            iterable = tqdm(pos_sample,desc='Finding Hot Pixels')
        else:
            iterable = pos_sample
        for pos in iterable:
            pos_md =  self.metadata.image_table[self.metadata.image_table.Position==pos]
            acqs = pos_md.acq.unique()
            if len(acqs)>n_acqs:
                acqs = random.sample(list(acqs),n_acqs)
            for acq in acqs:
                hot_pixel_dict[acq] = {}
                channels = pos_md[pos_md.acq==acq].Channel.unique()
                channels = set(list(channels)).intersection(self.channels)
                for channel in channels:
                    img = np.average(self.metadata.stkread(Position=pos,Channel=channel,acq=acq),axis=2)
                    bkg_sub = img-cv2.filter2D(img,-1,kernel)
                    avg = np.average(bkg_sub)
                    std = np.std(bkg_sub)
                    thresh = (avg+(std_thresh*std))
                    loc = np.where(bkg_sub>thresh)
                    X.extend(loc[0])
                    Y.extend(loc[1])
        img = np.histogram2d(X,Y,bins=2048,range=[[0,2048],[0,2048]])[0]
        loc = np.where(img>threshold_otsu(img))
        self.hotpixel_X = loc[:,0]
        self.hotpixel_Y = loc[:,1]
        pickle.dump(loc,open('/scratch/hotpixels.pkl','wb'))
