from tqdm import tqdm
from MERFISH_Objects.Position import *
from MERFISH_Objects.Classify import *
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
import dill as pickle

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
        if not os.path.exists(self.daemon_path):
            os.mkdir(self.daemon_path)
        self.position_daemon_path = os.path.join(self.daemon_path,'position')
        if not os.path.exists(self.position_daemon_path):
            os.mkdir(self.position_daemon_path)
            os.mkdir(os.path.join(self.position_daemon_path,'input'))
            os.mkdir(os.path.join(self.position_daemon_path,'output'))
        self.bitmap = self.merfish_config.bitmap
        self.channels = list(np.unique([channel for seq,hybe,channel in self.bitmap]))
        self.hybes = list(np.unique([hybe for seq,hybe,channel in self.bitmap]))
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
            
        self.completed = False
        self.passed = True
        
    def run(self):
        self.check_imaging()
        
    def check_imaging(self):
        self.metadata = Metadata(self.metadata_path)
        self.acqs = [i for i in self.metadata.image_table.acq.unique() if 'hybe' in i]
        self.posnames = self.metadata.image_table[self.metadata.image_table.acq.isin(self.acqs)].Position.unique()
        self.check_hot_pixel()
        self.check_flags()
        
    def check_hot_pixel(self):
        self.hotpixel = self.utilities.load_data(Dataset=self.dataset,Type='hot_pixels')
        if isinstance(self.hotpixel,type(None)):
            self.find_hot_pixels(std_thresh=self.parameters['std_thresh'],
                                 n_acqs=self.parameters['n_acqs'],
                                 kernel_size=self.parameters['hotpixel_kernel_size'])
            
    def check_flags(self):
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
            
    def find_hot_pixels(self,std_thresh=3,n_acqs=5,kernel_size=3):
        if kernel_size%2==0:
            kernel_size = kernel_size+1
        kernel = np.ones((kernel_size,kernel_size))
        kernel[int(kernel_size/2),int(kernel_size/2)] = 0
        kernel = kernel/np.sum(kernel)
        X = []
        Y = []
        hot_pixel_dict = {}
        if len(self.posnames)>self.n_pos:
            pos_sample = random.sample(list(self.posnames),self.n_pos)
        else:
            pos_sample = self.posnames
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
        # Need to Doube Check
        img = np.histogram2d(X,Y,bins=[img.shape[0],img.shape[1]],range=[[0,img.shape[0]],[0,img.shape[1]]])[0]
        loc = np.where(img>threshold_otsu(img))
        self.utilities.save_data(loc,Dataset=self.dataset,Type='hot_pixels')
