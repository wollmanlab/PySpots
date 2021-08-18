from metadata import Metadata
import pandas as pd
import argparse
import os
from cellpose import models
# from skimage.external import tifffile
from collections import Counter
import numpy as np
import multiprocessing
from functools import partial
import sys
import cv2
from tqdm import tqdm
from skimage import io
from fish_results import HybeData
from scipy.ndimage.morphology import distance_transform_edt as dte
# from skimage import morphology
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from fish_helpers import colorize_segmented_image
from skimage import filters
from skimage import morphology
from scipy import ndimage
from analysis_scripts.classify import spotcat 
from skimage.measure import regionprops
from PIL import Image
import torch
from scipy.ndimage import median_filter,gaussian_filter

import importlib
from MERFISH_Objects.FISHData import *
from metadata import Metadata
import os
from fish_helpers import *

class Segment_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.verbose = verbose
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.k = self.parameters['projection_k']
        self.channel = self.parameters['nucstain_channel']
        self.acq = 'infer'#self.parameters['nucstain_acq']
        self.acqname = self.parameters['nucstain_acq']
        self.projection_function = self.parameters['segment_projection_function']
        self.min_size = self.parameters['segment_min_size']
        self.overlap_threshold = self.parameters['segment_overlap_threshold']
        self.pixel_thresh = self.parameters['segment_pixel_thresh']
        self.z_thresh = self.parameters['segment_z_thresh']
        self.distance_thresh = self.parameters['segment_distance_thresh']
        self.model_type=self.parameters['segment_model_type']
        self.gpu = self.parameters['segment_gpu']
        self.batch_size = self.parameters['segment_batch_size']
        self.diameter = self.parameters['segment_diameter']
        self.channels = self.parameters['segment_channels']
        self.flow_threshold = self.parameters['segment_flow_threshold']
        self.cellprob_threshold = self.parameters['segment_cellprob_threshold']
        self.downsample = self.parameters['segment_downsample'] 
        self.two_dimensional = self.parameters['segment_two_dimensional']
        self.overwrite = self.parameters['segment_overwrite']
        self.singular_zindex = self.parameters['segment_singular_zindex']
        self.nuclear_blur = self.parameters['segment_nuclear_blur']
        self.pixel_size = self.parameters['segment_pixel_size']
        self.z_step_size = self.parameters['segment_z_step_size']

        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
            
        cellpose_inputs = {}
        cellpose_inputs['model_type'] = self.model_type
        cellpose_inputs['gpu'] = self.gpu
        cellpose_inputs['batch_size'] = self.batch_size
        cellpose_inputs['diameter'] = self.diameter
        cellpose_inputs['channels'] = self.channels
        cellpose_inputs['flow_threshold'] = self.flow_threshold
        cellpose_inputs['cellprob_threshold'] = self.cellprob_threshold
        self.cellpose_inputs = cellpose_inputs
        self.completed = False
        
    def run(self): 
        self.check_flags()
        self.find_nucstain()
        self.check_projection()
        self.check_cell_metadata()
        if self.overwrite:
            self.completed = False
        if not self.completed:
            self.generate_stk()
            self.initalize_cellpose()
            self.segment()
            if not self.two_dimensional:
                self.merge_labels_overlap('f')
                self.merge_labels_overlap('r')
            self.filter_labels()
            self.voronoi()
            self.generate_cell_metadata()
            self.update_flags()
            
    def check_flags(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Flags')]
        self.failed = False
        #Position
        flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                       posname=self.posname)
        if flag == 'Failed':
            log = self.posname+' Failed'
            self.completed = True
            self.failed = True
        # Segmentation
        flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                       posname=self.posname,channel=self.channel)
        if flag == 'Failed':
            log = 'Segmentation Failed'
            self.completed = True
            self.failed = True
            
        if self.failed:
            self.fishdata.add_and_save_data('Failed','flag',dataset=self.dataset,
                                                posname=self.posname,
                                                channel=self.channel)
            self.fishdata.add_and_save_data(log,'log',
                                                dataset=self.dataset,posname=self.posname,
                                                channel=self.channel)
        
    def find_nucstain(self):
        if self.acq == 'infer':
            self.acq = [i for i in os.listdir(self.metadata_path) if self.acqname in i][0]
        
    def check_projection(self):
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Projection Zindexes')]
        self.metadata = Metadata(os.path.join(self.metadata_path,self.acq))
        self.pos_metadata = self.metadata.image_table[(self.metadata.image_table.Position==self.posname)&(self.metadata.image_table.Channel==self.channel)]
        self.len_z = len(self.pos_metadata.Zindex.unique())
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
        if self.two_dimensional:
            self.zindexes = [0]
        self.nZ = len(self.zindexes)
        
        """ In future find beads for nucstain too """
        
    def check_cell_metadata(self):
        try:
            if self.two_dimensional:
                nuclei_mask = self.fishdata.load_data('nuclei_mask',
                                                  dataset=self.dataset,
                                                  posname=self.posname)
            else:
                nuclei_mask = self.fishdata.load_data('nuclei_mask',
                                                  dataset=self.dataset,
                                                  posname=self.posname,
                                                  zindex=self.zindexes[0])
        except:
            nuclei_mask = None
        if not isinstance(nuclei_mask,type(None)):
            self.update_flags()
            self.completed = True
    
    def project_image(self,sub_stk):
        if self.projection_function == 'max':
            img = sub_stk.max(axis=2)
        elif self.projection_function == 'mean':
            img = sub_stk.mean(axis=2)
        elif self.projection_function == 'median':
            img = sub_stk.median(axis=2)
        elif self.projection_function == 'sum':
            img = sub_stk.sum(axis=2)
        elif self.projection_function == 'None':
            img = sub_stk[:,:,0]
        return img
    
    def project_stk(self,stk):
        if self.verbose:
            iterable = tqdm(enumerate(self.zindexes),total=len(self.zindexes),desc='Projecting Nuclear Stack')
        else:
            iterable = enumerate(self.zindexes)
        # Need to be more flexible
        proj_stk = np.empty([stk.shape[0],stk.shape[1],len(self.zindexes)])
        self.translation_z = 0 # find beads in future
        for i,zindex in iterable:
            sub_zindexes = list(range(zindex-self.k+self.translation_z,zindex+self.k+self.translation_z+1))
            proj_stk[:,:,i] = self.project_image(stk[:,:,sub_zindexes])
        return proj_stk
    
    def normalize_image(self,image):
        image = image.astype(float)
        image = image-np.percentile(image.ravel(),0.001)
        image = image/np.percentile(image.ravel(),99.999)
        image[image<0]=0
        image[image>1]=1
        image = image*100000
        return image

    def process_image(self,image):
        image = image-gaussian_filter(image,self.nuclear_blur)
        image[image<0] = 0
        return image#np.log10(image+1)
    
    def process_stk(self,stk):
        if self.verbose:
            iterable = tqdm(range(stk.shape[2]),total=stk.shape[2],desc='Processing Stack')
        else:
            iterable = range(stk.shape[2])
        stk = stk.astype(float)
        bstk = stk.copy()
        for i in iterable:
            img = stk[:,:,i]
            bstk[:,:,i] = gaussian_filter(img,self.nuclear_blur)
        bsstk = stk-bstk
        bsstk[bsstk<0] = 0
        return np.log10(bsstk.mean(axis=2)+1)
    
    def generate_stk(self):
        stk = ''#np.empty([2048,2048,len(self.pos_metadata)])
        if self.verbose:
            iterable = tqdm(enumerate(self.pos_metadata.filename),total=len(self.pos_metadata),desc='Generating Nuclear Stack')
        else:
            iterable = enumerate(self.pos_metadata.filename)
        """ ensure these are in the right order"""
        for img_idx,fname in iterable:
            if isinstance(stk,str):
                img = cv2.imread(os.path.join(fname),-1)
                self.img_shape = img.shape
                stk = np.empty([self.img_shape[0],self.img_shape[1],len(self.pos_metadata)])
                stk[:,:,img_idx] = img
            else:
                stk[:,:,img_idx]=cv2.imread(os.path.join(fname),-1) # check which is faster
#             stk[:,:,img_idx]=io.imread(os.path.join(fname))
        if self.two_dimensional:
            self.nuclear_stack = self.process_stk(self.project_image(stk)[:,:,None])
            self.nuclear_images = [self.nuclear_stack]
        else:
            self.nuclear_stack = self.project_stk(stk)
            self.nuclear_images = self.stack_to_images(self.nuclear_stack)
        
    def initalize_cellpose(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Initialize Cellpose')]
        self.model = models.Cellpose(model_type=self.cellpose_inputs['model_type'],
                                     gpu=self.cellpose_inputs['gpu'])#,
#                                      batch_size=self.cellpose_inputs['batch_size'])
    def segment(self):
        if self.verbose:
            iterable = tqdm(self.nuclear_images,desc='Segmenting Nuclei Images')
        else:
            iterable = self.nuclear_images
        self.raw_mask_images = []
#         scale = int(2048*self.downsample)
        for image in iterable:
            image = self.process_image(image)
            image = self.normalize_image(image)
            if self.downsample!=1:
                image = np.array(Image.fromarray(image).resize((int(self.img_shape[1]*self.downsample),int(self.img_shape[0]*self.downsample)), Image.BICUBIC))
            raw_mask_image,flows,styles,diams = self.model.eval(image,
                                              diameter=self.cellpose_inputs['diameter']*self.downsample,
                                              channels=self.cellpose_inputs['channels'],
                                              flow_threshold=self.cellpose_inputs['flow_threshold'],
                                              cellprob_threshold=self.cellpose_inputs['cellprob_threshold'])
            if self.downsample!=1:
                 raw_mask_image = np.array(Image.fromarray(raw_mask_image).resize((self.img_shape[1],self.img_shape[0]), Image.NEAREST))
            self.raw_mask_images.append(raw_mask_image)
        self.mask_images = self.raw_mask_images
        self.mask_stack = self.images_to_stack(self.mask_images)
    
    def stack_to_images(self,stack):
        return [stack[:,:,z] for z in range(stack.shape[2])]

    def images_to_stack(self,images):
        return np.stack(images,axis=2)
    
    def merge_labels_overlap(self,order):
        """ Torch Speed up? """
        # Need a good way to ensure I am not merging cells
        Input = self.mask_images
        Output = [np.zeros([self.img_shape[0],self.img_shape[1]]) for i in range(self.nZ)]
        used_labels = 0
        if order == 'f':
            if self.verbose:
                iterable = tqdm(range(self.nZ),total=self.nZ,desc='Forward Merge Labels')
            else:
                iterable = range(self.nZ)
            start = 0
            step = 1
        elif order == 'r':
            if self.verbose:
                iterable = tqdm(reversed(range(self.nZ)),total=self.nZ,desc='Reverse Merge Labels')
            else:
                iterable = reversed(range(self.nZ))
            start = self.nZ-1
            step = -1
        for z in iterable:
            input_mask = Input[z]
            new_mask = np.zeros_like(input_mask)
            input_labels = np.unique(input_mask[input_mask>0].ravel())
            input_labels = input_labels[input_labels>0]
            if z==start:
                for input_label in input_labels[input_labels>0]:
                    input_label_mask = input_mask==input_label
                    input_size = np.sum(input_label_mask)
                    if input_size>self.min_size:
                        used_labels+=1
                        new_mask[input_label_mask]=used_labels
            else:
                output_mask = Output[z-step]
                output_labels = np.unique(output_mask.ravel())
                for input_label in input_labels:
                    input_label_mask = input_mask==input_label
                    input_size = np.sum(input_label_mask)
                    if input_size>=self.min_size:
                        overlap_labels = np.unique(output_mask[input_label_mask].ravel())
                        overlap_labels = overlap_labels[overlap_labels>0]
                        if len(overlap_labels)==0:
                            # doesnt match existing label make new one
                            used_labels+=1
                            new_mask[input_label_mask]=used_labels
                        else:
                            overlap = []
                            overlap_masks = []
                            for output_label in overlap_labels:
                                output_label_mask = output_mask==output_label
                                overlap_masks.append(output_label_mask)
                                output_size = np.sum(output_label_mask)
                                if output_size>=self.min_size:
                                    overlap.append(np.sum(output_label_mask&input_label_mask)/np.min([input_size,input_size]))
                            max_overlap = np.max(overlap)
                            if max_overlap>self.overlap_threshold:
                                overlap_label = overlap_labels[np.where(overlap==max_overlap)[0][0]]
                                new_mask[input_label_mask]=overlap_label
                            else:
                                used_labels+=1
                                new_mask[input_label_mask]=used_labels
            Output[z] = new_mask
        self.mask_images = Output
        self.mask_stack = self.images_to_stack(Output)
    
    def filter_labels(self):
        mask_stack = self.images_to_stack(self.mask_images)
        new_mask_stk = mask_stack.copy()
        if self.verbose:
            iterable = tqdm(np.unique(mask_stack[mask_stack>0].ravel()),desc='Filter Labels')
        else:
            iterable = np.unique(mask_stack[mask_stack>0].ravel())
        for cell in iterable:
            label_mask_stk = mask_stack==cell
            s = np.sum(np.sum(label_mask_stk,axis=0),axis=0)
            if np.sum(s)<self.pixel_thresh:
                new_mask_stk[label_mask_stk] = 0
            elif np.sum(s>0)<self.z_thresh:
                new_mask_stk[label_mask_stk] = 0
        self.mask_stack = new_mask_stk
        self.mask_images = self.stack_to_images(new_mask_stk)

    def voronoi(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Voronoi Segment')]
        inverted_binary_mask_stk = self.mask_stack==0
        distance_mask_stk = dte(inverted_binary_mask_stk,sampling=[self.pixel_size,self.pixel_size,self.z_step_size])
        max_mask_stk = distance_mask_stk<self.distance_thresh
#         labels = morphology.watershed(image=distance_mask_stk, markers=self.mask_stack,mask=max_mask_stk)
        labels = watershed(image=distance_mask_stk, markers=self.mask_stack,mask=max_mask_stk)
        self.voronoi_stack = labels
        self.voronoi_images = self.stack_to_images(labels)
        
    def generate_cell_metadata(self):
        metadata = []
        regions = regionprops(self.mask_stack)
        if self.verbose:
            iterable = tqdm(regions,total=len(regions),desc='Generating Cell Metadata')
        else:
            iterable = regions
        for region in iterable:
            cell_id = str(str(self.dataset)+'_'+str(self.posname)+'_cell_'+str(region.label))
            x,y,z = np.array(region.centroid).astype(int)
            nuclear_area = region.area
            total_area = np.sum(1*(self.voronoi_stack==region.label))
            metadata.append(pd.DataFrame([cell_id,x,y,z,nuclear_area,total_area,self.posname],index=['cell_id','x_pixel','y_pixel','z_index','nuclear_area','total_area','pos']).T)
        if len(metadata)>0:
            metadata = pd.concat(metadata,ignore_index=True)
        else:
            # maybe fail position here
            metadata = pd.DataFrame(columns=['cell_id','x_pixel','y_pixel','z_index','nuclear_area','total_area','pos'])
        self.cell_metadata = metadata
        self.save_masks()
        self.fishdata.add_and_save_data(self.cell_metadata,
                                        dtype='cell_metadata',
                                        dataset=self.dataset,
                                        posname=self.posname)
        
        
    def save_masks(self):
        if self.verbose:
            iterable = tqdm(enumerate(self.zindexes),total=len(self.zindexes),desc='Saving Masks')
        else:
            iterable = enumerate(self.zindexes)
        for i,z in iterable:
            if self.two_dimensional:
                self.fishdata.add_and_save_data(self.mask_images[0],
                                                dtype='nuclei_mask',
                                                dataset=self.dataset,
                                                posname=self.posname)
                self.fishdata.add_and_save_data(self.voronoi_images[0],
                                                dtype='cytoplasm_mask',
                                                dataset=self.dataset,
                                                posname=self.posname)
            else:
                self.fishdata.add_and_save_data(self.mask_images[i],
                                                dtype='nuclei_mask',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                zindex=z)
                self.fishdata.add_and_save_data(self.voronoi_images[i],
                                                dtype='cytoplasm_mask',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                zindex=z)
        self.update_flags()
        self.completed = True
            
            
    def view_mask(self,zindex,nuclei=True):
        Display(colorize_segmented_image(self.mask_stack[:,:,zindex]),rel_min=0,rel_max=100)
        
    def view_nucstain(self,zindex='mean',nuclei=True):
        if isinstance(zindex,str):
            temp = self.projection_function
            self.projection_function = zindex
            Display(self.project_image(self.mask_stack))
            self.projection_function = temp
        else:
            Display(self.mask_images[zindex])
        
    def update_flags(self):
        self.fishdata.add_and_save_data('Passed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            channel='DeepBlue')
        