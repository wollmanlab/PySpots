import os
import torch
import scipy
import pickle
import numpy as np
import trackpy as tp
from tqdm import tqdm
# from skimage import io
from itertools import repeat
from metadata import Metadata
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.ndimage.filters import convolve
from collections import Counter, defaultdict
from skimage.feature import match_template, peak_local_max
import importlib
from MERFISH_Objects.Utilities import *
from MERFISH_Objects.FISHData import *
from fish_results import HybeData
from datetime import datetime

from scipy.ndimage import median_filter,gaussian_filter
from skimage.registration import phase_cross_correlation


class Registration_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 hybe,
                 cword_config,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.hybe = hybe
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters  
        self.two_dimensional = self.parameters['two_dimensional']
        self.registration_threshold = self.parameters['registration_threshold']
        self.upsamp_factor = self.parameters['upsamp_factor']
        self.dbscan_eps = self.parameters['dbscan_eps']
        self.dbscan_min_samples = self.parameters['dbscan_min_samples']
        self.max_dist = self.parameters['max_dist']
        self.match_threshold = self.parameters['match_threshold']
        self.channel = self.parameters['registration_channel']
        self.hotpixel_kernel_size=self.parameters['hotpixel_kernel_size']
        self.image_blur_kernel = self.parameters['registration_image_blur_kernel']
        self.image_background_kernel = self.parameters['registration_image_background_kernel']
        self.utilities_path = self.parameters['utilities_path']
        self.ref_hybe = self.parameters['ref_hybe']
        self.subpixel_method = self.parameters['subpixel_method']
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.hotpixel = self.utilities.load_data(Dataset=self.dataset,Type='hot_pixels')
        self.hotpixel_X=self.hotpixel[0]
        self.hotpixel_Y=self.hotpixel[1]
        if self.hybe == self.ref_hybe:
            self.ref = True
        else:
            self.ref = False
        self.verbose=verbose
        
        self.fishdata_path = os.path.join(self.metadata_path,self.parameters['fishdata'])
        self.fishdata = FISHData(self.fishdata_path)
        
        self.completed = False
        self.passed = True
        
    def run(self):
        self.check_flags()
        self.main()
        
    def main(self):
        if not self.completed:
            if self.parameters['registration_method'] =='image':
                self.image_registration()
            else:
                self.check_beads()
    
    
    def update_user(self,message):
        """ For User Display"""
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
    
    def check_flags(self):
        """ Check flags to ensure this code should be executed"""
        flag = self.utilities.load_data(Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Channel=self.channel,
                                        Type='flag')
        if flag=='Failed': # Self Failed
            self.completed = True
        elif flag=='Passed':
            self.completed = True
        else:
            flag = self.utilities.load_data(Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Type='flag')
            if flag=='Failed': # Hybe Failed
                self.completed = True
                self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Channel=self.channel,
                                             Type='flag')
                self.utilities.save_data(str(self.hybe+' Failed'),
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Channel=self.channel,
                                             Type='log')
            else:
                flag = self.utilities.load_data(Dataset=self.dataset,
                                        Position=self.posname,
                                        Type='flag')
                if flag=='Failed': # Position Failed
                    self.completed = True
                    self.report_failure(str(self.posname+' Failed'))                   
                    self.utilities.save_data(str(self.posname+' Failed'),
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Channel=self.channel,
                                             Type='log')
                    self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Channel=self.channel,
                                             Type='flag')
                    self.utilities.save_data(str(self.posname+' Failed'),
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Type='log')
                    self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Type='flag')
                # else:
                #     if self.parameters['registration_method'] =='image':
                #         self.image_registration()
                #     else:
                #         self.check_beads()
            
    def load_image(self,hybe,channel):
        """ Load Image for image based registration"""
        ### Need to move from hardcode
        acq = [i for i in os.listdir(self.metadata_path) if hybe+'_' in i][0]
        temp_metadata = Metadata(os.path.join(self.metadata_path,acq))
        try:
            stk = temp_metadata.stkread(Position=self.posname,Channel=channel,hybe=hybe)
            
        except:
            """ Issue with imaging"""
            stk = None
            self.completed = True
            self.utilities.save_data('Imaging Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Channel=self.channel,
                                     Type='log')
            self.utilities.save_data('Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Channel=self.channel,
                                     Type='flag')
            self.utilities.save_data('Registration Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Type='log')
            self.utilities.save_data('Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Type='flag')
            self.utilities.save_data(str(self.hybe)+' Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Type='log')
            self.utilities.save_data('Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Type='flag')
        image = stk.mean(axis=2)
        denoised = gaussian_filter(image,self.image_blur_kernel)
        background = gaussian_filter(denoised,self.image_background_kernel)
        image = image.astype(float)-background.astype(float)
        zscore = image-np.median(image)
        zscore = zscore/np.std(image)
        return zscore
        
    def image_registration(self):
        """ Register using cross correlation"""
        tforms = self.fishdata.load_data('tforms',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        if not isinstance(tforms,type(None)):
            self.tforms = tforms
            self.completed = True
            self.utilities.save_data('Passed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Channel=self.channel,
                                     Type='flag')
        else:
            if self.verbose:
                self.update_user('Registering Image')
            self.translation_z = 0
            self.residual = 0
            self.nbeads = 0
            if self.hybe==self.parameters['ref_hybe']:
                self.translation_x = 0
                self.translation_y = 0
            else:
                self.ref_image = self.load_image(self.parameters['ref_hybe'],'FarRed')
                self.image = self.load_image(self.hybe,'FarRed')
                shift, error, diffphase = phase_cross_correlation(self.ref_image,
                                                                  self.image,
                                                                  upsample_factor=10)
                self.translation_x = shift[1]
                self.translation_y = shift[0]
            self.save_tforms()
            
    def check_tforms(self):
        tforms = self.fishdata.load_data('tforms',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        if not isinstance(tforms,type(None)):
            self.tforms = tforms
            self.completed = True
            self.utilities.save_data('Passed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Channel=self.channel,
                                     Type='flag')
        else:
            self.find_tforms()
                
    def check_beads(self):
        beads = self.fishdata.load_data('beads',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        if self.parameters['registration_overwrite']:
            beads = None
        if not isinstance(beads,type(None)):
            self.beads = beads
            if len(self.beads)==0:
                self.report_failure('not enough beads found')
            else:
                self.check_tforms()
        else:
            self.find_beads()
            
    def process_image(self,img):
        img = self.remove_hotpixels(img)
        bkg = gaussian_filter(img,self.image_background_kernel)
        img = img-bkg
        img = gaussian_filter(img,self.image_blur_kernel)
        """ Zscore Image"""
        temp = np.percentile(img.ravel(),[25,50,75])
        img = img-temp[1]
        img = img/(temp[2]-temp[1])
        img[img<0] = 0 # below median likely not a bead
        return img
    
    def create_hotpixel_kernel(self):
        """ Create the kernel that will correct a hotpixel"""
        if self.verbose:
            self.update_user('Creating Hot Pixel Kernel')
        kernel = np.ones((self.hotpixel_kernel_size,self.hotpixel_kernel_size))
        kernel[int(self.hotpixel_kernel_size/2),int(self.hotpixel_kernel_size/2)] = 0
        self.hotpixel_kernel = kernel/np.sum(kernel)
        
    def remove_hotpixels(self,img):
        """ Replace hot pixels with the mean of their neighbors"""
        img[self.hotpixel_X,self.hotpixel_Y] = convolve(img,self.hotpixel_kernel)[self.hotpixel_X,self.hotpixel_Y] 
        return img
        
    def load_stack(self):
        self.metadata = Metadata(self.metadata_path)
        try:
            self.stk = self.metadata.stkread(Position=self.posname,Channel=self.channel,hybe=self.hybe).astype(float)
        except:
            """ Issue with imaging"""
            self.stk = None
            self.completed = True
            self.utilities.save_data('Imaging Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Channel=self.channel,
                                     Type='log')
            self.utilities.save_data('Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Channel=self.channel,
                                     Type='flag')
            self.utilities.save_data('Registration Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Type='log')
            self.utilities.save_data('Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Hybe=self.hybe,
                                     Type='flag')
            self.utilities.save_data(str(self.hybe)+' Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Type='log')
            self.utilities.save_data('Failed',
                                     Dataset=self.dataset,
                                     Position=self.posname,
                                     Type='flag')
        if self.verbose:
            self.update_user('Loading Stack')
        # Should add a way to exclude cells from bead find
        # Filter Out Low Frequency Background
        # Filter Out High Frequency Noise
        self.create_hotpixel_kernel()
        if self.verbose:
            iterable = tqdm(range(self.stk.shape[2]),desc='Processing Stack')
        else:
            iterable = range(self.stk.shape[2])
        if self.hybe=='nucstain':
            """ mask out cells before calling beads """
            img = self.stk.mean(2)
            """ Robust Zscore"""
            temp = np.percentile(img.ravel(),[25,50,75])
            img = img-temp[1]
            img = img/(temp[2]-temp[0])
            """ Binarize"""
            mask = img>1 # Greater than 75th percentile  # MOVE
            """ Remove Small Objects (beads) """
            from skimage.morphology import remove_small_objects
            mask = remove_small_objects(mask,200) # MOVE
            """ dialate """
            """ may have issues if very dense cells """
            from scipy.ndimage.morphology import binary_dilation
            mask = binary_dilation(mask,iterations=10) # MOVE
        for i in iterable:
            self.stk[:,:,i] =  self.process_image(self.stk[:,:,i])
        if self.hybe=='nucstain':
            self.stk[mask,:] = 0
        # Threshold to prevent False Positive Bead Calls
        # thresh = np.percentile(self.stk.ravel(),99.9)
        # self.stk = self.stk-thresh
        # self.stk[self.stk<0] = 0
        # # Blur to Ensure Clean Center
        # for i in range(self.stk.shape[2]):
        #     self.stk[:,:,i] = gaussian_filter(self.stk[:,:,i],2)
        if self.two_dimensional:
            self.stk = self.stk.mean(axis=2)

    def generate_template(self):
        #Create Bead Template
        # Remove Hard Code
        bead = np.zeros((7, 7, 5))
        bead[3, 3, 2] = 1
        bead = gaussian(bead, (1.5, 1.5, 0.85))
        Ave_Bead = bead/bead.max()
        self.bead_template = Ave_Bead
        self.upsamp_bead = resize(self.bead_template[2:5, 2:5, 1:4],
                             (3*self.upsamp_factor, 3*self.upsamp_factor, 3*self.upsamp_factor),
                             mode='constant',anti_aliasing=False)
        if self.two_dimensional:
            self.bead_template = self.bead_template.mean(axis=2)
            self.upsamp_bead = self.upsamp_bead.mean(axis=2)
            
        
    def find_beads(self):
        """
        2D/3D registration from sparse bead images.
        Parameters
        ----------
        fnames_dict : dict
            Dictionary (hybe name:list filenames) to load bead images
        bead_template : numpy.array
            Array of normalized values of 'average bead' intensities.
            This helps to not pick up noisy spots/hot pixels.
        ref_reference : str - default('hybe1')
            Which hybe is the reference destination to map source hybes onto.
        max_dist : int - default(50)
            The maximum distanced allowed between beads found when pairing 
            beads between multiple hybes.
        match_threshold : float - default(0.75)
            The similarity threshold between bead_template and images to 
            consider potential beads.
        Returns
        -------
        tforms : dict
            Dictionary of translation vectors (x, y, z)? maybe (y, x, z)
        """

        self.load_stack()
        # Find Beads in 2D First
        img = self.stk.mean(2) # Mean is more robust to noise and out of focus light
        """ Zscore Image"""
        temp = np.percentile(img.ravel(),[25,50,75])
        img = img-temp[1]
        img = img/(temp[2]-temp[1])
        # img[img<0] = 0
        img[img<2] = 2 # must be more than 4 std from mean to be considered bead
        # thresh = np.percentile(img.ravel(),90)
        # img[img<thresh] = thresh
        self.features = tp.locate(img,
                                  minmass=100,
                                  diameter=(15,15),
                                  separation=20) # Move from hard code
        
        subpixel_beads = []
        if self.subpixel_method == 'template':
            if self.verbose:
                iterable = tqdm(self.features.shape[0],desc='Finding Subpixel Centers')
            else:
                iterable = ref_beads
            ref_beads = np.zeros([self.features.shape[0],2],dtype=int)
            ref_beads[:,0] = self.features.y
            ref_beads[:,1] = self.features.x
            # ref_beads = peak_local_max(img)#,threshold_abs=np.percentile(img.ravel(),99.9))
            # Find Beads in 3D
            z = torch.tensor(self.stk[ref_beads[:,0],ref_beads[:,1],:]).max(1).indices.numpy()
            ref_beads = np.concatenate([ref_beads.T,z[:,None].T]).T
            ref_beads = ref_beads.astype(int)
            ref_beads = [ref_beads[i,:] for i in range(ref_beads.shape[0])]
            self.generate_template()
            if self.two_dimensional:
                for y, x in iterable:
                    substk = self.stk[y-5:y+6, x-5:x+6]
                    if substk.shape[0] != 11 or substk.shape[1] != 11:
                        continue # candidate too close to edge
                    try:
                        upsamp_substk = resize(substk,
                                               (substk.shape[0]*self.upsamp_factor,
                                                substk.shape[1]*self.upsamp_factor),
                                                mode='constant',anti_aliasing=False)
                    except:
                        continue
                    bead_match = match_template(upsamp_substk,
                                                self.upsamp_bead, pad_input=True)
                    yu, xu = np.where(bead_match==bead_match.max())
                    yu = (yu[0]-int(upsamp_substk.shape[0]/2))/self.upsamp_factor
                    xu = (xu[0]-int(upsamp_substk.shape[1]/2))/self.upsamp_factor
                    ys, xs = (yu+y, xu+x)
                    subpixel_beads.append((ys, xs,0))
            else:
                for y, x, z in iterable:
                    substk = self.stk[y-5:y+6, x-5:x+6, z-2:z+3]
                    if substk.shape[0] != 11 or substk.shape[1] != 11:
                        continue # candidate too close to edge
                    try:
                        upsamp_substk = resize(substk,
                                               (substk.shape[0]*self.upsamp_factor,
                                                substk.shape[1]*self.upsamp_factor,
                                                substk.shape[2]*self.upsamp_factor),
                                                mode='constant',anti_aliasing=False)
                    except:
                        continue
                    bead_match = match_template(upsamp_substk,
                                                self.upsamp_bead, pad_input=True)
                    yu, xu, zu = np.where(bead_match==bead_match.max())
                    yu = (yu[0]-int(upsamp_substk.shape[0]/2))/self.upsamp_factor
                    xu = (xu[0]-int(upsamp_substk.shape[1]/2))/self.upsamp_factor
                    zu = (zu[0]-int(upsamp_substk.shape[2]/2))/self.upsamp_factor
                    ys, xs, zs = (yu+y, xu+x, zu+z)
                    subpixel_beads.append((ys, xs, zs))
                    
        elif self.subpixel_method == 'max':
            window = 5
            subpixel_beads = []
            if self.verbose:
                iterable = tqdm(self.features.iterrows(),total=self.features.shape[0],desc='Finding Subpixel Centers')
            else:
                iterable = self.features.iterrows()
            for i,row in iterable:
                x = int(row.x)
                y = int(row.y)
                if self.two_dimensional:
                    substk = self.stk[y-window:y+window+1, x-window:x+window+1]
                    if substk.shape[0] != 2*window+1 or substk.shape[1] != 2*window+1:
                        continue # candidate too close to edge
                    try:
                        upsamp_substk = resize(substk,
                                               (substk.shape[0]*self.upsamp_factor,
                                                substk.shape[1]*self.upsamp_factor),
                                                mode='constant',anti_aliasing=False)
                    except:
                        continue
                    yu = np.argmax(gaussian_filter(np.array([upsamp_substk[d,:,:].mean() for d in range(upsamp_substk.shape[0])]),5))
                    xu = np.argmax(gaussian_filter(np.array([upsamp_substk[:,d,:].mean() for d in range(upsamp_substk.shape[1])]),5))
                    yu = (yu-int(upsamp_substk.shape[0]/2))/self.upsamp_factor
                    xu = (xu-int(upsamp_substk.shape[1]/2))/self.upsamp_factor
                    ys, xs, zs = (yu+y, xu+x)
                else:
                    substk = self.stk[y-window:y+window+1, x-window:x+window+1,:]
                    if substk.shape[0] != 2*window+1 or substk.shape[1] != 2*window+1:
                        continue # candidate too close to edge
                    try:
                        upsamp_substk = resize(substk,
                                               (substk.shape[0]*self.upsamp_factor,
                                                substk.shape[1]*self.upsamp_factor,
                                                substk.shape[2]*self.upsamp_factor),
                                                mode='constant',anti_aliasing=False)
                    except:
                        continue
                    yu = np.argmax(gaussian_filter(np.array([upsamp_substk[d,:,:].mean() for d in range(upsamp_substk.shape[0])]),5))
                    xu = np.argmax(gaussian_filter(np.array([upsamp_substk[:,d,:].mean() for d in range(upsamp_substk.shape[1])]),5))
                    zu = np.argmax(gaussian_filter(np.array([upsamp_substk[:,:,d].mean() for d in range(upsamp_substk.shape[2])]),5))
                    yu = (yu-int(upsamp_substk.shape[0]/2))/self.upsamp_factor
                    xu = (xu-int(upsamp_substk.shape[1]/2))/self.upsamp_factor
                    zu = zu/self.upsamp_factor
                    if zu==0:
                        continue
                    ys, xs, zs = (yu+y, xu+x, zu)
                    subpixel_beads.append((ys, xs, zs))
        
        self.beads = subpixel_beads
        # del self.stk
        if len(self.beads)==0:
            self.report_failure('not enough beads found')
        else:
            self.save_beads()
            self.find_tforms()

    def find_tforms(self):
        if self.ref:
            self.generate_reference_hybe()
        else:
            self.calculate_transformation()
            
    def generate_reference_hybe(self):
        if self.verbose:
            self.update_user('Calculating Transformation')
        self.ref_beadarray = np.stack(self.beads,axis=0)
        self.ref_tree = KDTree(self.ref_beadarray[:, :2])
        self.db_clusts = DBSCAN(min_samples=self.dbscan_min_samples, eps=self.dbscan_eps)
        self.utilities.save_data(self.ref_beadarray,Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='ref_beadarray')
        self.utilities.save_data(self.ref_tree,Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='ref_tree')
        self.utilities.save_data(self.db_clusts,Dataset=self.dataset,Position=self.posname,Hybe=self.hybe,Type='db_clusts')
        self.translation_x = 0
        self.translation_y = 0
        self.translation_z = 0
        self.residual = 0
        self.nbeads = self.ref_beadarray.shape[0]
        if self.nbeads<self.dbscan_min_samples:
            self.report_failure('Not enough reference beads found.')
        else:
            self.save_tforms()
            
    def report_failure(self,log):
        self.utilities.save_data(log,
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Channel=self.channel,
                                 Type='log')
        self.utilities.save_data('Failed',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Channel=self.channel,
                                 Type='flag')
        self.utilities.save_data('Registration Failed',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Type='log')
        self.utilities.save_data('Failed',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Type='flag')
        self.utilities.save_data(str(self.hybe)+' Failed',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Type='log')
        self.utilities.save_data('Failed',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Type='flag')

        
    def load_ref(self):
        beads = self.fishdata.load_data('beads',dataset=self.dataset,posname=self.posname,hybe=self.ref_hybe)
        if not isinstance(beads,type(None)):
            self.ref_beadarray = np.stack(beads,axis=0)
            self.ref_tree = KDTree(self.ref_beadarray[:, :2])
            self.db_clusts = DBSCAN(min_samples=self.dbscan_min_samples, eps=self.dbscan_eps)
            self.utilities.save_data(self.ref_beadarray,Dataset=self.dataset,Position=self.posname,Hybe=self.ref_hybe,Type='ref_beadarray')
            self.utilities.save_data(self.ref_tree,Dataset=self.dataset,Position=self.posname,Hybe=self.ref_hybe,Type='ref_tree')
            self.utilities.save_data(self.db_clusts,Dataset=self.dataset,Position=self.posname,Hybe=self.ref_hybe,Type='db_clusts')
            self.reg_proceed = True
        else:
            self.reg_proceed = False
            
    def calculate_transformation(self):
        """
        Calculate transformation to register 2 sets of images together
        Given a set of candidate bead coordinates (xyz) and a reference hybe find min-error translation.

        This task is trivial given a paired set of coordinates for source/destination beads. 
        The input to this function is unpaired so the first task is pairing beads. Given the 
        density of beads in relation to the average distance between beads it is not reliable to 
        simply use closest-bead-candidate pairing. However, for each bead in the destination we can find 
        all beads within from distance in the source set of beads and calculate the difference of all these pairs.
        Bead pairs that are incorrect have randomly distributed differences however correct bead pairs all 
        have very similar differences. So a density clustering of the differences is performed to identify 
        the set of bead pairings representing true bead pairs between source/destination.

        The best translation is found my minimizing the mean-squared error between source/destination after pairing.
        """
        self.ref_beadarray = self.utilities.load_data(Dataset=self.dataset,Position=self.posname,Hybe=self.ref_hybe,Type='ref_beadarray')
        self.ref_tree = self.utilities.load_data(Dataset=self.dataset,Position=self.posname,Hybe=self.ref_hybe,Type='ref_tree')
        self.db_clusts = self.utilities.load_data(Dataset=self.dataset,Position=self.posname,Hybe=self.ref_hybe,Type='db_clusts')
        if isinstance(self.ref_beadarray,type(None)):
            if self.verbose:
                self.update_user('ref_beadarray is none')
            self.load_ref()
        elif isinstance(self.ref_tree,type(None)):
            if self.verbose:
                self.update_user('ref_tree is none')
            self.load_ref()
        elif isinstance(self.db_clusts,type(None)):
            if self.verbose:
                self.update_user('db_clusts is none')
            self.load_ref()
        else:
            self.reg_proceed = True
        if not self.reg_proceed:
            self.load_ref()
            # Add a try to load reference beads and build these objects
        if self.reg_proceed:
            self.beadarray = np.stack(self.beads, axis=0)
            t_est = []
            ref_beads = []
            dest_beads = []
            close_beads = self.ref_tree.query_ball_point(self.beadarray[:, :2], r=self.max_dist)
            if self.verbose:
                iterable = tqdm(zip(close_beads,self.beadarray),total=self.beadarray.shape[0],desc='Calculating Transformation')
            else:
                iterable = zip(close_beads, self.beadarray)
            for i, bead in iterable:
                if len(i)==0:
                    continue
                for nbor in i:
                    t = self.ref_beadarray[nbor]-bead
                    t_est.append(np.subtract(self.ref_beadarray[nbor], bead))
                    ref_beads.append(self.ref_beadarray[nbor])
                    dest_beads.append(bead)
            ref_beads = np.array(ref_beads)
            dest_beads = np.array(dest_beads)
            if len(t_est)<self.dbscan_min_samples:
                if self.verbose:
                    self.update_user('Not enough reference beads found.')
                self.passed = False
                self.utilities.save_data('Not enough reference beads found.',
                                        Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Channel=self.channel,
                                        Type='log')
                self.utilities.save_data('Failed',
                                        Dataset=self.dataset,
                                        Position=self.posname,
                                        Hybe=self.hybe,
                                        Channel=self.channel,
                                        Type='flag')
                self.utilities.save_data('Registration Failed',
                                         Dataset=self.dataset,
                                         Position=self.posname,
                                         Hybe=self.hybe,
                                         Type='log')
                self.utilities.save_data('Failed',
                                         Dataset=self.dataset,
                                         Position=self.posname,
                                         Hybe=self.hybe,
                                         Type='flag')
                self.utilities.save_data(str(self.hybe)+' Failed',
                                         Dataset=self.dataset,
                                         Position=self.posname,
                                         Type='log')
                self.utilities.save_data('Failed',
                                         Dataset=self.dataset,
                                         Position=self.posname,
                                         Type='flag')
            if self.passed:
                t_est = np.stack(t_est, axis=0)
                self.db_clusts.fit(t_est)
                most_frequent_cluster = Counter(self.db_clusts.labels_)
                most_frequent_cluster.pop(-1)
                try:
                    most_frequent_cluster = most_frequent_cluster.most_common(1)[0][0]
                except IndexError:
                    if self.verbose:
                        self.update_user('Index Error')
                    self.passed = False
                    self.utilities.save_data('Not enough reference beads found.',
                                            Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Type='log')
                    self.utilities.save_data('Failed',
                                            Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Type='flag')
                    self.utilities.save_data('Registration Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Type='log')
                    self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Type='flag')
                    self.utilities.save_data(str(self.hybe)+' Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Type='log')
                    self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Type='flag')
            if self.passed:
                paired_beads_idx = self.db_clusts.labels_==most_frequent_cluster
                ref = ref_beads[paired_beads_idx]
                dest = dest_beads[paired_beads_idx]
                t_est = t_est[paired_beads_idx]
                def error_func(translation):
                    fit = np.add(translation, dest)
                    fit_error = np.sqrt(np.subtract(ref, fit)**2)
                    fit_error = np.mean(fit_error)
                    return fit_error
                # Optimize translation to map paired beads onto each other
                opt_t = scipy.optimize.fmin(error_func, np.mean(t_est, axis=0), full_output=True, disp=False)
                if opt_t[1]>self.registration_threshold:
                    self.error = 'Residual too high'
                    if self.verbose:
                        self.update_user(str(self.error))
                    self.passed = False
                    self.utilities.save_data('Residual too high '+str(opt_t[1]),
                                            Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Type='log')
                    self.utilities.save_data('Failed',
                                            Dataset=self.dataset,
                                            Position=self.posname,
                                            Hybe=self.hybe,
                                            Channel=self.channel,
                                            Type='flag')
                    self.utilities.save_data('Registration Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Type='log')
                    self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Hybe=self.hybe,
                                             Type='flag')
                    self.utilities.save_data(str(self.hybe)+' Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Type='log')
                    self.utilities.save_data('Failed',
                                             Dataset=self.dataset,
                                             Position=self.posname,
                                             Type='flag')
                self.translation_x = opt_t[0][1]
                self.translation_y = opt_t[0][0]
                self.translation_z = opt_t[0][2]
                self.residual = opt_t[1]
                self.nbeads = sum(paired_beads_idx)
                self.save_tforms()
            
    def load_beads(self):
        self.beads = self.fishdata.load_data('beads',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        
    def save_beads(self):
        self.fishdata.add_and_save_data(self.beads,'beads',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        
    def save_tforms(self):
        self.tforms = {'x':self.translation_x,'y':self.translation_y,'z':self.translation_z}
        self.fishdata.add_and_save_data(self.tforms,'tforms',dataset=self.dataset,posname=self.posname,hybe=self.hybe)
        self.completed = True
        self.utilities.save_data('Passed',
                                 Dataset=self.dataset,
                                 Position=self.posname,
                                 Hybe=self.hybe,
                                 Channel=self.channel,
                                 Type='flag')
        