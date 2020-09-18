from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from tensorflow.python.client import device_lib
from skimage import restoration
from tqdm import tqdm
import importlib
import os
from MERFISH_Objects.Utilities import *
import numpy as np
import tifffile
import warnings
from MERFISH_Objects.FISHData import *
from metadata import Metadata
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import sys
from functools import partial


def normalize_image(img,normalization_rel_min=50,normalization_rel_max=95,normalization_max=1000):
    if np.sum(img.ravel())==0:
        return img
    else:
        img = img-np.percentile(img.ravel(),normalization_rel_min)
        img = img/np.percentile(img.ravel(),normalization_rel_max)
        img = img*normalization_max
    return img
        
class Deconvolution_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 hybe,
                 channel,
                 cword_config,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.channel = channel
        self.hybe = hybe
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.psf_dict = self.merfish_config.psf_dict_3d
        self.deconvolution_niterations=self.parameters['deconvolution_niterations']
        self.deconvolution_batches=self.parameters['deconvolution_batches']
        self.gpu=self.parameters['deconvolution_gpu']
        self.utilities_path = self.parameters['utilities_path']
        self.normalization_max = self.parameters['normalization_max']
        self.normalization_rel_max = self.parameters['normalization_rel_max']
        self.normalization_rel_min = self.parameters['normalization_rel_min']
        self.floor = self.parameters['floor']
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        self.verbose = verbose
        
        self.stack_loaded = False
        self.completed = False
        self.passed = True
        
    def run(self):
        self.check_flags()
            
    def check_flags(self):
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.proceed = True
        flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex='all')
        if flag=='Failed': # Self Failed
            self.completed = True
        elif flag=='Passed':
            self.completed = True
        else:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
            if flag=='Failed': # Channel Failed
                self.completed = True
                self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                self.fishdata.add_and_save_data(str(self.channel)+' Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
            elif flag=='Passed':
                self.completed = True
            else:
                flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
                if flag=='Failed': # Hybe Failed
                    self.completed = True
                    self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel,zindex='all')
                    self.fishdata.add_and_save_data(str(self.channel)+' Failed','log',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel,zindex='all')
                    self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel)
                    self.fishdata.add_and_save_data(str(self.hybe)+' Failed','log',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel)
                elif flag=='Passed': #Hybe Failed
                    self.completed = True
                else:
                    flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname)
                    if flag=='Failed': # Position Failed
                        self.completed = True
                        self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel,zindex='all')
                        self.fishdata.add_and_save_data(str(self.channel)+' Failed','log',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel,zindex='all')
                        self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel)
                        self.fishdata.add_and_save_data(str(self.hybe)+' Failed','log',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe,channel=self.channel)
                        self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe)
                        self.fishdata.add_and_save_data(str(self.posname)+' Failed','log',dataset=self.dataset,
                                                    posname=self.posname,hybe=self.hybe)
                    elif flag=='Passed':
                        self.completed = True
                    else:
                        self.check_image_flags()
            
    def main(self):
        self.load_stack()
        self.deconvolve()
        self.normalize()
        if self.floor:
            self.stk[self.stk<0]=0
        self.stk = self.stk.astype('uint16')
        self.save_stack()
        
    def check_image_flags(self):
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
                # Not sure what to do here yet
        if len(self.not_started)==0: # All Images have been started
            if len(self.started)==0: # All Images have been completed
                self.main()
        
    def load_stack(self):
        if self.verbose:
            iterable = tqdm(enumerate(self.zindexes),desc='Loading Stack')
        else:
            iterable = enumerate(self.zindexes)
        stk = np.zeros([2048,2048,len(self.zindexes)])
        for i,z in iterable:
            img = self.fishdata.load_data('image',
                                          dataset=self.dataset,
                                          posname=self.posname,
                                          hybe=self.hybe,
                                          channel=self.channel,
                                          zindex=z)
            if isinstance(img,type(None)):
                pass
            else:
                stk[:,:,i] = img
        self.stk = stk
        self.stack_loaded = True
        
    def save_stack(self):
        if self.verbose:
            iterable = tqdm(enumerate(self.zindexes),desc='Saving Stack')
        else:
            iterable = enumerate(self.zindexes)
        for i,z in iterable:
            img = self.stk[:,:,i]
            self.fishdata.add_and_save_data(img,'image',
                                          dataset=self.dataset,
                                          posname=self.posname,
                                          hybe=self.hybe,
                                          channel=self.channel,
                                          zindex=z)
            
        self.completed = True
        self.passed = True
        self.fishdata.add_and_save_data('Passed',
                                        'flag',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.channel,
                                        zindex='all')
        self.fishdata.add_and_save_data('Passed',
                                        'flag',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.channel)
        
    def deconvolve(self):
        self.load_psf()
        stk = self.stk.astype('float64')
        stk = stk-stk.min()
        stk = stk/stk.max()
        if self.verbose:
            iterable = tqdm(range(int(round(self.deconvolution_batches/2))),desc='Deconvolving Stack')
        else:
            iterable = range(int(round(self.deconvolution_batches/2)))
        step = int(stk.shape[0]/(self.deconvolution_batches/2))
        if self.gpu:
            #fd_restoration.RichardsonLucyDeconvolver(3).initialize()
            if self.deconvolution_batches>1:
                for i in iterable:
                    i0 = int(step*i)
                    for j in iterable:
                        j0 = int(step*j)
                        temp = stk[i0:i0+step,j0:j0+step,:]
                        if self.gpu:
                            stk[i0:i0+step,j0:j0+step,:] = self.gpu_algorithm.run(fd_data.Acquisition(data=temp, kernel=self.psf), niter=self.deconvolution_niterations).data
                        else:
                            stk[i0:i0+step,j0:j0+step,:] = restoration.richardson_lucy(temp, self.psf,self.deconvolution_niterations, clip=False)
            else:
                warnings.filterwarnings("ignore")
                stk = self.gpu_algorithm.run(fd_data.Acquisition(data=stk, kernel=self.psf), niter=self.deconvolution_niterations).data
                warnings.filterwarnings("default")
        else:
            stk = restoration.richardson_lucy(stk, self.psf,self.deconvolution_niterations, clip=False)
        self.stk = stk
        del self.gpu_algorithm
        
    def load_psf(self):
        # Need to update psf with projection skip
        self.psf = self.psf_dict[self.channel]
        
    def normalize(self):
        if self.verbose:
            iterable = tqdm(range(self.stk.shape[2]),desc='Normalizing Stack')
        else:
            iterable = range(self.stk.shape[2])
        Input = [self.stk[:,:,z] for z in range(self.stk.shape[2])]
        pfunc = partial(normalize_image,
                        normalization_rel_min=self.normalization_rel_min,
                        normalization_rel_max=self.normalization_rel_max,
                        normalization_max=self.normalization_max)
        with multiprocessing.Pool(10) as ppool:
            sys.stdout.flush()
            for z,img in enumerate(ppool.imap(pfunc,Input)):
                self.stk[:,:,z] = img
            ppool.close()
            sys.stdout.flush()
#         for z in iterable: #Parralelize this 
#             self.stk[:,:,z] = self.normalize_image(self.stk[:,:,z])
            
    def normalize_image(self,img):
        if np.sum(img.ravel())==0:
            return img
        else:
            img = img-np.percentile(img.ravel(),self.normalization_rel_min)
            img = img/np.percentile(img.ravel(),self.normalization_rel_max)
            img = img*self.normalization_max
            return img
        
    def check_projection(self):
        if self.verbose:
            print('Checking Projection Zindexes')
        self.metadata = Metadata(self.metadata_path)
        self.len_z = len(self.metadata.image_table[(self.metadata.image_table.Position==self.posname)].Zindex.unique())
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