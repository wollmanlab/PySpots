from hybescope_config.microscope_config import *
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter
from scipy.ndimage.filters import convolve
from scipy import interpolate
from tqdm import tqdm
import importlib
from metadata import Metadata
from MERFISH_Objects.Utilities import *
from MERFISH_Objects.FISHData import *
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from skimage import restoration
from skimage.registration import phase_cross_correlation
from datetime import datetime
import trackpy as tp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Image_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 hybe,
                 channel,
                 zindex,
                 cword_config,
                 verbose=False):
        """Class to Process Raw Images

        Args:
            metadata_path (str):  Path to Raw Data
            dataset (str): Name of Dataset
            posname (str): Name of Position
            hybe (str): Name of Hybe
            channel (str): Name of Channel
            zindex (str): Name of Zindex
            cword_config (str): Name of Config Module
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.channel = channel
        self.hybe = hybe
        self.zindex = int(zindex)
        self.verbose = verbose
        
        self.acq = [i for i in os.listdir(self.metadata_path) if self.hybe+'_' in i.lower()][-1]
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters =  self.merfish_config.parameters
        self.psf_dict = self.merfish_config.psf_dict
        self.deconvolution_niterations=self.parameters['deconvolution_niterations']
        self.deconvolution_batches=self.parameters['deconvolution_batches']
        self.gpu=self.parameters['deconvolution_gpu']
        self.two_dimensional = self.parameters['two_dimensional']
        
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        
        self.hotpixel = self.utilities.load_data(Dataset=self.dataset,Type='hot_pixels')
        self.chromatic_dict = self.merfish_config.chromatic_dict
        self.projection_function=self.parameters['projection_function']
        self.dtype=self.parameters['dtype']
        self.background_kernel=self.parameters['background_kernel']
        self.blur_kernel=self.parameters['blur_kernel']
        self.background_method=self.parameters['background_method']
        self.blur_method=self.parameters['blur_method']
        self.hotpixel_kernel_size=self.parameters['hotpixel_kernel_size']
        self.utilities_path = self.parameters['utilities_path']
        self.overwrite = False
        self.proceed = True
        self.completed = False
        self.passed = True
        self.daemon_path = self.parameters['daemon_path']
        self.image_daemon_path = os.path.join(self.daemon_path,'image')
        self.img = None
        if not os.path.exists(self.image_daemon_path):
            os.mkdir(self.image_daemon_path)
            os.mkdir(os.path.join(self.image_daemon_path,'input'))
            os.mkdir(os.path.join(self.image_daemon_path,'output'))
        
    def run(self):
        if self.parameters['image_overwrite']:
            self.main()
        else:
            self.check_flags()
            
    def check_flags(self):
        """FIX Move to Class"""
        self.proceed = True
        self.failed = False
        # Position
        flag = self.utilities.load_data(Dataset=self.dataset,
                                        Position=self.posname,
                                        Type='flag')
        if flag == 'Failed':
            log = self.posname+' Failed'
            self.completed = True
            self.failed = True
        # Hybe
        if not self.failed:
            flag = self.utilities.load_data(Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Type='flag')
            if flag == 'Failed':
                log = self.hybe+' Failed'
                self.completed = True
                self.failed = True
        # Channel
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
        # Zindex
        if not self.failed:
            flag = self.utilities.load_data(Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Zindex=self.zindex,
                                            Type='flag')
            if flag == 'Failed':
                log = self.zindex+' Failed'
                self.completed = True
                self.failed = True
        if self.failed:
            self.utilities.save_data('Failed',
                                    Dataset=self.dataset,
                                    Position=self.posname,
                                    Hybe=self.hybe,
                                    Channel=self.channel,
                                    Zindex=self.zindex,
                                    Type='flag')
            self.utilities.save_data(log,
                                    Dataset=self.dataset,
                                    Position=self.posname,
                                    Hybe=self.hybe,
                                    Channel=self.channel,
                                    Zindex=self.zindex,
                                    Type='log')
        else:
            """ Check Spots"""
            try:
                """ Change to check if file exists """
                self.spots = self.fishdata.load_data('spotcalls',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                hybe=self.hybe,
                                                channel=self.channel,
                                                zindex=self.zindex)
            except:
                self.spots = None
            if not isinstance(self.spots,type(None)): # Spots already found
                self.completed = True
                self.proceed = False
                """ Only pass if len spots>0 FIX"""
                self.utilities.save_data('Passed',
                                        Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Channel=self.channel,
                                        Zindex=self.zindex,
                                        Type='flag')
            else:
                """ Check Image """
                try:
                    """ Change to check if file exists """
                    self.img = self.fishdata.load_data('image',
                                                    dataset=self.dataset,
                                                    posname=self.posname,
                                                    hybe=self.hybe,
                                                    channel=self.channel,
                                                    zindex=self.zindex)
                except:
                    self.img = None
                self.main()
            
            
    def main(self):
        self.load_data()
        if self.proceed:
            if isinstance(self.img,type(None)):
                self.processed_sub_stk = np.zeros_like(self.sub_stk)
                for i in range(self.processed_sub_stk.shape[2]):
                    self.img = self.sub_stk[:,:,i]
                    self.load_chromatic()
                    self.remove_hotpixels()
                    self.register_image_xy()
                    self.subtract_background()
                    if self.deconvolution_niterations>0:
                        self.deconvolve()
                    self.smooth()
                    self.sub_stk[:,:,i] = self.img
                self.project()
                self.save_data()
                   
   
    def update_user(self,message):
        """ For User Display"""
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def load_data(self):
        if self.overwrite:
            if self.verbose:
                self.update_user('Overwriting Processing')
            self.proceed = True
        if self.proceed:
            """ Load Metadata """
            if self.verbose:
                self.update_user('Loading Metadata')
            self.metadata = Metadata(os.path.join(self.metadata_path,self.acq))
            self.imaged_zindexes = self.metadata.image_table.Zindex.unique()
            
            """ Load Transformations """
            if self.verbose:
                self.update_user('Loading Transformation')
            self.translation = self.fishdata.load_data('tforms',
                                                       dataset=self.dataset,
                                                       posname=self.posname,
                                                       hybe=self.hybe)
            if isinstance(self.translation,type(None)):
                self.proceed = False
            else:
                self.translation_x = self.translation['x']
                self.translation_y = self.translation['y']
                self.translation_z = int(round(self.translation['z']))
            
                """ Calculate Zindexes """
                self.k = self.parameters['projection_k']
                if self.two_dimensional:
                    self.zindexes = [0]
                else:
                    self.zindexes = np.array(list(range(self.zindex-self.k+self.translation_z,self.zindex+self.k+self.translation_z+1)))
                    self.zindexes = list(self.zindexes[np.isin(self.zindexes,self.imaged_zindexes)])
                    if len(self.zindexes)==0:
                        self.proceed = False
                        print('No Zindexes')
                # Might be issues if the z transformation is too large
                if self.proceed:
                    """ Loading Images """
                    if self.verbose:
                        self.update_user('Loading Sub Stack')
                    try:
                        if self.two_dimensional:
                            """ Use all zindexes"""
                            self.sub_stk = self.metadata.stkread(Position=self.posname,
                                                                Channel=self.channel,
                                                                verbose=self.verbose).astype(self.dtype)
                        else:
                            """ Use some zindexes"""
                            self.sub_stk = []
                            for z in self.zindexes:
                                self.sub_stk.append(self.metadata.stkread(Position=self.posname,
                                                                Channel=self.channel,
                                                                Zindex=z,
                                                                verbose=self.verbose).astype(self.dtype).max(2))
                            self.sub_stk = np.dstack(self.sub_stk)

                    except Exception as e:
                        print(e)
                        # Translation in z too large for this z index
                        # Just use an average image for this position
                        # Zeros may be an issue here
                        """ use the minimum of all zindexes"""
                        if self.verbose:
                            self.update_user('Using min of image')
                        try:
                            self.sub_stk = self.metadata.stkread(Position=self.posname,hybe=self.hybe,Channel=self.channel,verbose=self.verbose).astype(self.dtype)
                            self.sub_stk = np.min(self.sub_stk,axis=2)[:,:,None]
                        except:
                            print('Likely this channel wasnt imaged')
                            print(self.posname,self.hybe,self.channel)
                            self.proceed = False
                    if len(self.sub_stk.shape)==2:
                        self.sub_stk = self.sub_stk[:,:,None]
                    self.len_y,self.len_x,self.len_z = self.sub_stk.shape

    def project(self):
        """ Wrapper """
        """ Take a 3D Volume and make it 2D"""
        """ Helps with Autofocus Differences"""
        if self.verbose:
            self.update_user('Projecting Sub Stack')
        if len(self.sub_stk.shape)==2:
            self.img = self.sub_stk
        elif len(self.sub_stk.shape)==3:
            self.img = self.project_image(self.sub_stk)
        else:
            raise(ValueError('Sub_stk can only by 2 or 3d',self.sub_stk.shape))
        del self.sub_stk
        
    def project_image(self,sub_stk):
        """ Take a 3D Volume and make it 2D"""
        """ Helps with Autofocus Differences"""
        if self.projection_function == 'max':
            img = sub_stk.max(axis=2)
        elif self.projection_function == 'mean':
            img = sub_stk.mean(axis=2)
        elif self.projection_function == 'median':
            img = sub_stk.median(axis=2)
        elif self.projection_function == 'sum':
            img = sub_stk.sum(axis=2)
        else:
            img = sub_stk.mean(axis=2)
        return img
    
    def load_chromatic(self):
        """ Correct for Chromatic Differences"""
        """ Not Implemented since Monochromatic"""
        if self.verbose:
            self.update_user('Loading Chromatic')
        if len(np.unique([c for r,h,c in self.merfish_config.bitmap]))>1:
            # NEED TO CALCULATE #
            self.chromatic_x = self.chromatic_dict[self.channel]['x']
            self.chromatic_y = self.chromatic_dict[self.channel]['y']
        else:
            # Not using chromatic since monochromatic
            self.chromatic_x = np.array(range(self.len_x))
            self.chromatic_y = np.array(range(self.len_y))
        
    def create_hotpixel_kernel(self):
        """ Create the kernel that will correct a hotpixel"""
        if self.verbose:
            self.update_user('Creating Hot Pixel Kernel')
        kernel = np.ones((self.hotpixel_kernel_size,self.hotpixel_kernel_size))
        kernel[int(self.hotpixel_kernel_size/2),int(self.hotpixel_kernel_size/2)] = 0
        self.hotpixel_kernel = kernel/np.sum(kernel)
        
    def remove_hotpixels(self):
        """ Replace hot pixels with the mean of their neighbors"""
        self.create_hotpixel_kernel()
        if self.verbose:
            self.update_user('Correcting Hot Pixels')
        self.hotpixel_X=self.hotpixel[0]
        self.hotpixel_Y=self.hotpixel[1]
        self.img[self.hotpixel_X,self.hotpixel_Y] = convolve(self.img,self.hotpixel_kernel)[self.hotpixel_X,self.hotpixel_Y]    
        
    def register_image_xy(self):
        """ warp image to correct for chromatic and translation"""
        if self.verbose:
            self.update_user('Correcting Chromatic and Translaton')
        i2 = interpolate.interp2d(self.chromatic_x+self.translation_x, self.chromatic_y+self.translation_y, self.img,fill_value=None)
        self.img = i2(range(self.len_x), range(self.len_y))
        
    def blur(self,method,kernel):
        """ Blur Image"""
        if method =='None':
            blur = np.zeros_like(self.img)
        elif kernel==0:
            blur = self.img
        elif method =='median':
            blur = median_filter(self.img,kernel)
        elif method == 'gaussian':
            blur = gaussian_filter(self.img,kernel)
        else:
            raise(ValueError(method,'is not an implemented method'))
        return blur
    
    def subtract_background(self):
        """ Calculate and Subtract Background """
        """ Remove Noise """
        self.img = self.blur('gaussian',self.blur_kernel) # Move from hard code
        
        """ Remove signal larger than expected"""
        if self.background_method!='None':
            if self.verbose:
                self.update_user('Subtracting Background')
            self.bkg = self.blur(self.background_method,self.background_kernel)
            # Subtract or divide here?
            self.img = self.img.astype(float) - self.bkg.astype(float)
            self.img[self.img<0] = 0 # Prevents going past zero
            self.img = self.img.astype(self.dtype)
            del self.bkg
            
    def deconvolve(self):
        """ Deconvolve with calculated Point Spread Function"""
        self.psf = self.psf_dict[self.channel] 
        img = self.img.astype(float)
        # preserve scale
        img_min = img.min()
        img = img-img_min
        img_max = img.max()
        img = img/img_max
        img = img+10**-4 # Cant have 0
        if self.verbose:
            self.update_user('Deconvolution')
        if self.gpu:
            self.gpu_algorithm = fd_restoration.RichardsonLucyDeconvolver(2).initialize()
            img = self.gpu_algorithm.run(fd_data.Acquisition(data=img, kernel=self.psf), niter=self.deconvolution_niterations).data
            del self.gpu_algorithm
        else:
            img = restoration.richardson_lucy(img, self.psf,self.deconvolution_niterations, clip=False)
        # restore scale
        img = img-10**-4
        img = img*img_max
        img = img+img_min
        self.img = img
        
    def smooth(self):
        """ Same as Blur"""
        if self.verbose:
            self.update_user('Smoothing Image')
        if self.blur_kernel>0:
            self.img = self.blur(self.blur_method,self.blur_kernel)
            
    def convert_image_bit(self):
        """ Fixes issue with converting bits and going pass max/min"""
        if self.verbose:
            self.update_user('Converting Image Dtype')
        # Add a gain here to use more of the dynamic range 
        self.img = self.img*self.parameters['gain']
        # Add code here to prevent lapping past min to max
        self.img[self.img<=np.iinfo(self.dtype).min] = np.iinfo(self.dtype).min
        self.img[self.img>=np.iinfo(self.dtype).max] = np.iinfo(self.dtype).max
        self.img = self.img.astype(self.dtype)
        
    def call_spots(self):
        """ Find Spots in Images """
        """ ZScore Image"""
        zscore = self.img.copy()
        zscore = zscore-np.percentile(zscore.ravel(),25)
        zscore = zscore/np.percentile(zscore.ravel(),75)
        # zscore[zscore<4] = 0
        
        """ Detect Spots"""
        self.spots = tp.locate(zscore,
                               self.parameters['spot_diameter'],
                               minmass=self.parameters['spot_minmass'],
                               separation=self.parameters['spot_separation'])
        # self.spots = tp.locate(zscore,
        #                        self.parameters['spot_diameter'],
        #                        percentile=self.parameters['spot_minmass'],
        #                        separation=self.parameters['spot_separation']) 
        
        """ UPDATE CHECK IF THERE ARE NO SPOTS AND ALERT USER """
        if len(self.spots)==0:
            self.utilities.save_data('No Spots Detected',
                                    Dataset=self.dataset,
                                    Position=self.posname,
                                    Hybe=self.hybe,
                                    Channel=self.channel,
                                    Zindex=self.zindex,
                                    Type='log')
            
    def save_data(self):
        """ Save Data using FISHDATA"""
        if self.verbose:
            self.update_user('Saving Data')
        self.convert_image_bit()
        self.fishdata.add_and_save_data(self.img,
                                        'image',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.channel,
                                        zindex=self.zindex)
        if self.parameters['image_call_spots']:
            self.fishdata.add_and_save_data(self.spots,
                                            'spotcalls',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe=self.hybe,
                                            channel=self.channel,
                                            zindex=self.zindex)
        self.utilities.save_data('Passed',
                                    Dataset=self.dataset,
                                    Position=self.posname,
                                    Hybe=self.hybe,
                                    Channel=self.channel,
                                    Zindex=self.zindex,
                                    Type='flag')
        self.completed = True
        
