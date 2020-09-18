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
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters =  self.merfish_config.parameters
        self.hotpixel_X=self.merfish_config.hotpixel_X
        self.hotpixel_Y=self.merfish_config.hotpixel_Y
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
        
        self.completed = False
        self.passed = True
        
    def run(self):
        self.check_flags()
            
    def check_flags(self):
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.proceed = True
        self.img = self.fishdata.load_data('image',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=self.hybe,
                                        channel=self.channel,
                                        zindex=self.zindex)
        if not isinstance(self.img,type(None)): # Image already done
            self.completed = True
        else:
            flag = self.fishdata.load_data('flag',
                                           dataset=self.dataset,
                                           posname=self.posname,
                                           hybe=self.hybe,
                                           channel=self.channel,
                                           zindex=self.zindex)
            if flag=='Failed': #Self Failed
                self.completed = True
            elif flag=='Passed':
                self.completed = True
            else:
                flag = self.fishdata.load_data('flag',
                                               dataset=self.dataset,
                                               posname=self.posname,
                                               hybe=self.hybe,
                                               channel=self.channel)
                if flag=='Failed': #Channel Failed
                    self.completed = True
                    self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=self.zindex)
                    self.fishdata.add_and_save_data(str(self.channel)+ 'Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=self.zindex)
                elif flag=='Passed':
                    self.completed = True
                else:
                    flag = self.fishdata.load_data('flag',
                                                   dataset=self.dataset,
                                                   posname=self.posname,
                                                   hybe=self.hybe)
                    if flag=='Failed': # Hybe Failed
                        self.completed = True
                        self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=self.zindex)
                        self.fishdata.add_and_save_data(str(self.channel)+ 'Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=self.zindex)
                        self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                        self.fishdata.add_and_save_data(str(self.hybe)+ 'Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                    elif flag=='Passed':
                        self.completed = True
                    else:
                        flag = self.fishdata.load_data('flag',
                                                       dataset=self.dataset,
                                                       posname=self.posname)
                        if flag=='Failed': # Position Failed
                            self.completed = True
                            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=self.zindex)
                            self.fishdata.add_and_save_data(str(self.channel)+ 'Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel,zindex=self.zindex)
                            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                            self.fishdata.add_and_save_data(str(self.hybe)+ 'Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe,channel=self.channel)
                            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
                            self.fishdata.add_and_save_data(str(self.position)+ 'Failed','log',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
                        elif flag=='Passed':
                            self.completed = True
                        else:
                            self.main()
            
    def main(self):
        self.load_data()
        self.project()
        self.load_chromatic()
        self.create_hotpixel_kernel()
        self.convert_image_bit()
        self.remove_hotpixels()
        self.register_image_xy()
        self.subtract_background()
        self.smooth()
        self.save_data()
        
    def load_data(self):
        if self.verbose:
            print('Loading Metadata')
        self.utilities = Utilities_Class(self.utilities_path)
        self.metadata = Metadata(self.metadata_path)
        #Load Tforms
        if self.verbose:
            print("Loading Transformation")
        self.translation = self.fishdata.load_data('tforms',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        self.translation_x = self.translation['x']
        self.translation_y = self.translation['y']
        self.translation_z = round(self.translation['z'])
        self.k = self.parameters['projection_k']
        zindexes = list(range(self.zindex-self.k+self.translation_z,self.zindex+self.k+self.translation_z+1))
        # Might be issues if the z transformation is too large
        if self.verbose:
            print('Loading Sub Stack')
        try:
            self.sub_stk = self.metadata.stkread(Position=self.posname,hybe=self.hybe,Channel=self.channel,Zindex=zindexes,verbose=self.verbose).astype(self.dtype)
        except:
            # Translation in z too large for this z index
            # Just use an average image for this position
            # Zeros may be an issue here
            self.sub_stk = self.metadata.stkread(Position=self.posname,hybe=self.hybe,Channel=self.channel,verbose=self.verbose).astype(self.dtype)
            self.sub_stk = np.min(self.sub_stk,axis=2)
            
    def project(self):
        if self.verbose:
            print('Projecting Sub Stack')
        if len(self.sub_stk.shape)==2:
            self.img = self.sub_stk
        elif len(self.sub_stk.shape)==3:
            self.img = self.project_image(self.sub_stk)
        else:
            raise(ValueError('Sub_stk can only by 2 or 3d',self.sub_stk.shape))
        self.len_x,self.len_y = self.img.shape
        del self.sub_stk
        
    def project_image(self,sub_stk):
        if self.projection_function == 'max':
            img = sub_stk.max(axis=2)
        elif self.projection_function == 'mean':
            img = sub_stk.mean(axis=2)
        elif self.projection_function == 'median':
            img = sub_stk.median(axis=2)
        return img
    
    def load_chromatic(self):
        if self.verbose:
            print('Loading Chromatic')
        self.chromatic_x = self.chromatic_dict[self.channel]['x']
        self.chromatic_y = self.chromatic_dict[self.channel]['y']
        
    def create_hotpixel_kernel(self):
        if self.verbose:
            print('Creating Hot Pixel Kernel')
        kernel = np.ones((self.hotpixel_kernel_size,self.hotpixel_kernel_size))
        kernel[int(self.hotpixel_kernel_size/2),int(self.hotpixel_kernel_size/2)] = 0
        self.hotpixel_kernel = kernel/np.sum(kernel)
        
    def convert_image_bit(self):
        if self.verbose:
            print('Converting Image Dtype')
        self.img = self.img.astype(self.dtype)
        # Add code here to prevent lapping past min to max
        
    def remove_hotpixels(self):
        if self.verbose:
            print('Correcting Hot Pixels')
        self.img[self.hotpixel_X,self.hotpixel_Y] = convolve(self.img,self.hotpixel_kernel)[self.hotpixel_X,self.hotpixel_Y]
        
    def register_image_xy(self):
        if self.verbose:
            print('Correcting Chromatic and Translaton')
        i2 = interpolate.interp2d(self.chromatic_x+self.translation_x, self.chromatic_y+self.translation_y, self.img)
        self.img = i2(range(self.len_x), range(self.len_y))
        
    def blur(self,method,kernel):
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
        if self.background_method!='None':
            if self.verbose:
                print('Subtracting Background')
            self.bkg = self.blur(self.background_method,self.background_kernel)
            self.bkg[self.bkg>self.img] = self.img[self.bkg>self.img]
            self.img = self.img-self.bkg
            del self.bkg
            
    def smooth(self):
        if self.verbose:
            print('Smoothing Image')
        if self.blur_kernel>0:
            self.img = self.blur(self.blur_method,self.blur_kernel)
    
    def save_data(self):
        if self.verbose:
            print('Saving Image')
        # Use FISHData instead
        self.fishdata.add_and_save_data(self.img.astype(self.dtype),
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
        
