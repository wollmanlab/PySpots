from hybescope_config.microscope_config import *
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter
from scipy.ndimage.filters import convolve
from scipy import interpolate
from tqdm import tqdm
import importlib
from metadata import Metadata
from MERFISH_Objects.Utilities import *
from fish_results import HybeData
from MERFISH_Objects.FISHData import *
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from skimage import restoration
from skimage.registration import phase_cross_correlation
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
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.channel = channel
        self.hybe = hybe
        self.zindex = int(zindex)
        
        self.acq = [i for i in os.listdir(self.metadata_path) if self.hybe+'_' in i][0]
        
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
        self.hotpixel_X=self.hotpixel[0]#self.merfish_config.hotpixel_X
        self.hotpixel_Y=self.hotpixel[1]#self.merfish_config.hotpixel_Y
        self.chromatic_dict = self.merfish_config.chromatic_dict
        self.projection_function=self.parameters['projection_function']
        self.dtype=self.parameters['dtype']
        self.background_kernel=self.parameters['background_kernel']
        self.blur_kernel=self.parameters['blur_kernel']
        self.background_method=self.parameters['background_method']
        self.blur_method=self.parameters['blur_method']
        self.hotpixel_kernel_size=self.parameters['hotpixel_kernel_size']
        self.utilities_path = self.parameters['utilities_path']
        self.verbose = verbose
        self.proceed = True
        self.completed = False
        self.passed = True
        self.daemon_path = self.parameters['daemon_path']
        self.image_daemon_path = os.path.join(self.daemon_path,'image')
        if not os.path.exists(self.image_daemon_path):
            os.mkdir(self.image_daemon_path)
            os.mkdir(os.path.join(self.image_daemon_path,'input'))
            os.mkdir(os.path.join(self.image_daemon_path,'output'))
        
    def run(self):
        self.check_flags()
            
    def check_flags(self):
        """Update"""
        self.proceed = True
        self.failed = False
        # Position
        flag = self.fishdata.load_data('flag',
                                           dataset=self.dataset,
                                           posname=self.posname)
        if flag == 'Failed':
            log = self.posname+' Failed'
            self.completed = True
            self.failed = True
        # Hybe
        if not self.failed:
            flag = self.fishdata.load_data('flag',
                                           dataset=self.dataset,
                                           posname=self.posname,
                                           hybe=self.hybe)
            if flag == 'Failed':
                log = self.hybe+' Failed'
                self.completed = True
                self.failed = True
        # Channel
        if not self.failed:
            flag = self.fishdata.load_data('flag',
                                           dataset=self.dataset,
                                           posname=self.posname,
                                           hybe=self.hybe,
                                           channel=self.channel)
            if flag == 'Failed':
                log = self.channel+' Failed'
                self.completed = True
                self.failed = True
        # Zindex
        if not self.failed:
            flag = self.fishdata.load_data('flag',
                                           dataset=self.dataset,
                                           posname=self.posname,
                                           hybe=self.hybe,
                                           channel=self.channel,
                                           zindex=self.zindex)
            if flag == 'Failed':
                log = self.zindex+' Failed'
                self.completed = True
                self.failed = True
        if self.failed:
            self.fishdata.add_and_save_data('Failed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe=self.hybe,
                                            channel=self.channel,
                                            zindex=self.zindex)
            self.fishdata.add_and_save_data(log,'log',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe=self.hybe,
                                            channel=self.channel,
                                            zindex=self.zindex)
        else:
            try:
                self.img = self.fishdata.load_data('image',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                hybe=self.hybe,
                                                channel=self.channel,
                                                zindex=self.zindex)
            except:
                self.img = None
            if not isinstance(self.img,type(None)): # Image already done
                self.completed = True
                self.fishdata.add_and_save_data('Passed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe=self.hybe,
                                            channel=self.channel,
                                            zindex=self.zindex)
            else:
                self.main()
            
    def main(self):
        self.load_data()
        if self.proceed:
            self.project()
            self.load_chromatic()
#             self.convert_image_bit()
            self.remove_hotpixels()
            self.register_image_xy()
            self.subtract_background()
            if self.deconvolution_niterations>0:
                self.deconvolve()
            self.smooth()
#             if not self.hybe=='nucstain':
#                 self.subtract_autofluorescence()
            self.save_data()
        
    def load_data(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Loading Metadata')]
#         if not self.hybe=='nucstain':
#             self.check_autofluorescence() # Needs to be done first
        if self.proceed:
            # Load Metadata
            self.metadata = Metadata(os.path.join(self.metadata_path,self.acq))
            # Load Tforms
            if self.verbose:
                i = [i for i in tqdm([],desc='Loading Transformation')]
            self.translation = self.fishdata.load_data('tforms',
                                                       dataset=self.dataset,
                                                       posname=self.posname,
                                                       hybe=self.hybe)
#             if self.hybe=='nucstain':
#                 self.translation = {'x':0,'y':0,'z':0}
            self.translation_x = self.translation['x']
            self.translation_y = self.translation['y']
            self.translation_z = int(round(self.translation['z']))
            self.k = self.parameters['projection_k']
            if self.two_dimensional:
                zindexes = [0]
            else:
                zindexes = list(range(self.zindex-self.k+self.translation_z,self.zindex+self.k+self.translation_z+1))
            # Might be issues if the z transformation is too large
            if self.verbose:
                i = [i for i in tqdm([],desc='Loading Sub Stack')]
            try:
                if self.two_dimensional:
                    """ Use all zindexes"""
                    self.sub_stk = self.metadata.stkread(Position=self.posname,
                                                         Channel=self.channel,
                                                         verbose=self.verbose).astype(self.dtype)
                else:
                    """ Use some zindexes"""
                    self.sub_stk = self.metadata.stkread(Position=self.posname,
                                                         Channel=self.channel,
                                                         Zindex=zindexes,
                                                         verbose=self.verbose).astype(self.dtype)

            except:
                # Translation in z too large for this z index
                # Just use an average image for this position
                # Zeros may be an issue here
                """ use the minimum of all zindexes"""
                try:
                    self.sub_stk = self.metadata.stkread(Position=self.posname,hybe=self.hybe,Channel=self.channel,verbose=self.verbose).astype(self.dtype)
                    self.sub_stk = np.min(self.sub_stk,axis=2)
                except:
                    print('Likely this channel wasnt imaged')
                if self.verbose:
                    i = [i for i in tqdm([],desc='Using min of image')]
                    
    def check_autofluorescence(self):
        """ Load Acquired Autofluorescence"""
        """ Or Initialize Image Class to make one"""
        
        flag = self.fishdata.load_data('flag',
                                           dataset=self.dataset,
                                           posname=self.posname,
                                           hybe='nucstain',
                                           channel=self.channel,
                                           zindex=self.zindex)
        self.autofluorescence = self.fishdata.load_data('image',
                                           dataset=self.dataset,
                                           posname=self.posname,
                                           hybe='nucstain',
                                           channel=self.channel,
                                           zindex=self.zindex)
        if flag == 'Passed':
            self.proceed = True
        elif flag == 'Started':
            self.proceed = False
        elif flag == 'Failed':
            if self.verbose:
                i = [i for i in tqdm([],desc='Error with Autofluorescence')]
            self.proceed = False
            """ Error Out """
            self.completed = True
            self.passed = False
        else:
            self.proceed = False
            
#             """ Start the AF processing """
#             fname = self.dataset+'_'+self.posname+'_'+'nucstain'+'_'+self.channel+'_'+str(self.zindex)+'.pkl'
#             fname_path = os.path.join(self.image_daemon_path,'input',fname)
#             data = {'metadata_path':self.metadata_path,
#                         'dataset':self.dataset,
#                         'posname':self.posname,
#                         'hybe':'nucstain',
#                         'channel':self.channel,
#                         'zindex':str(self.zindex),
#                         'cword_config':self.cword_config,
#                         'level':'image'}
#             pickle.dump(data,open(fname_path,'wb'))
#             self.flag = self.fishdata.add_and_save_data('Started','flag',
#                                                             dataset=self.dataset,
#                                                             posname=self.posname,
#                                                             hybe='nucstain',
#                                                             channel=self.channel,
#                                                             zindex=str(self.zindex))

    def project(self):
        """ Wrapper """
        """ Take a 3D Volume and make it 2D"""
        """ Helps with Autofocus Differences"""
        if self.verbose:
            i = [i for i in tqdm([],desc='Projecting Sub Stack')]
        if len(self.sub_stk.shape)==2:
            self.img = self.sub_stk
        elif len(self.sub_stk.shape)==3:
            self.img = self.project_image(self.sub_stk)
        else:
            raise(ValueError('Sub_stk can only by 2 or 3d',self.sub_stk.shape))
        self.len_x,self.len_y = self.img.shape
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
            i = [i for i in tqdm([],desc='Loading Chromatic')]
        if len(np.unique([c for r,h,c in self.merfish_config.bitmap]))>1:
            self.chromatic_x = self.chromatic_dict[self.channel]['x']
            self.chromatic_y = self.chromatic_dict[self.channel]['y']
        else:
            # Not using chromatic since monochromatic
            self.chromatic_x = np.array(range(self.len_x))#self.chromatic_dict[self.channel]['x']
            self.chromatic_y = np.array(range(self.len_y))#self.chromatic_dict[self.channel]['y']
        
    def create_hotpixel_kernel(self):
        """ Create the kernel that will correct a hotpixel"""
        if self.verbose:
            i = [i for i in tqdm([],desc='Creating Hot Pixel Kernel')]
        kernel = np.ones((self.hotpixel_kernel_size,self.hotpixel_kernel_size))
        kernel[int(self.hotpixel_kernel_size/2),int(self.hotpixel_kernel_size/2)] = 0
        self.hotpixel_kernel = kernel/np.sum(kernel)
        
    def remove_hotpixels(self):
        """ Replace hot pixels with the mean of their neighbors"""
        self.create_hotpixel_kernel()
        if self.verbose:
            i = [i for i in tqdm([],desc='Correcting Hot Pixels')]
        self.img[self.hotpixel_X,self.hotpixel_Y] = convolve(self.img,self.hotpixel_kernel)[self.hotpixel_X,self.hotpixel_Y]    
        
    def register_image_xy(self):
        """ warp image to correct for chromatic and translation"""
        if self.verbose:
            i = [i for i in tqdm([],desc='Correcting Chromatic and Translaton')]
        i2 = interpolate.interp2d(self.chromatic_x+self.translation_x, self.chromatic_y+self.translation_y, self.img)
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
        # Remove Noise
        self.img = self.blur('gaussian',0.9) # Move from hard code
        # Remove signal larger than expected
        if self.background_method!='None':
            if self.verbose:
                i = [i for i in tqdm([],desc='Subtracting Background')]
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
            i = [i for i in tqdm([],desc='Deconvolution')]
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
            i = [i for i in tqdm([],desc='Smoothing Image')]
        if self.blur_kernel>0:
            self.img = self.blur(self.blur_method,self.blur_kernel)
    
    def subtract_autofluorescence(self):
        """ Correct for Autofluorescence"""
        if self.verbose:
            i = tqdm([],desc='Correcting Autofluorescence')
        if isinstance(self.autofluorescence,type(None)):
            if self.verbose:
                i = tqdm([],desc='Autofluorescence Image empty')
        else:
            """ Could have an issue if AF changes due to bleaching"""
            self.img = self.img.astype(float)-self.autofluorescence.astype(float)
            self.img[self.img<0] = 0 # Prevents wrapping past 0
            self.img = self.img.astype(self.dtype)
            
    def convert_image_bit(self):
        """ Fixes issue with converting bits and going pass max/min"""
        if self.verbose:
            i = [i for i in tqdm([],desc='Converting Image Dtype')]
        # Add code here to prevent lapping past min to max
        self.img[self.img<=np.iinfo(self.dtype).min] = np.iinfo(self.dtype).min
        self.img[self.img>=np.iinfo(self.dtype).max] = np.iinfo(self.dtype).max
        self.img = self.img.astype(self.dtype)
        
    def save_data(self):
        """ Save Data using FISHDATA"""
        if self.verbose:
            i = [i for i in tqdm([],desc='Saving Image')]
        self.convert_image_bit()
        self.fishdata.add_and_save_data(self.img,
                                        'image',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.channel,
                                        zindex=self.zindex)
        self.fishdata.add_and_save_data('Passed',
                                        'flag',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.channel,
                                        zindex=self.zindex)
        self.completed = True
        
