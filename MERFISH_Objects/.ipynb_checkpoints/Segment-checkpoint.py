#!/usr/bin/env python

from metadata import Metadata
import pandas as pd
import argparse
import os
from cellpose import models
from skimage.external import tifffile
from collections import Counter
import numpy as np
import multiprocessing
from functools import partial
import sys
from tqdm import tqdm
from skimage import io
from fish_results import HybeData
from scipy.ndimage.morphology import distance_transform_edt as dte
from skimage import morphology
import matplotlib.pyplot as plt
from fish_helpers import colorize_segmented_image
from skimage import filters
from skimage import morphology
from scipy import ndimage
from analysis_scripts.classify import spotcat 
from skimage.measure import regionprops
from MERFISH_Objects.FISHData import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str, help="path to dataset /usr/project/dataset/")
    parser.add_argument("-n","--ncpu", type=int, dest="ncpu", default=10, action='store', help="how many threads to use for multiprocessing")
    parser.add_argument("-na","--nuclear_acq", type=str, dest="nuclear_acq", default='infer', action='store', help="name of nuclear stain acquisition")
    parser.add_argument("-v","--verbose", type=bool, dest="verbose", default=False, action='store', help="print stements")
    parser.add_argument("-ms","--min_size", type=int, dest="min_size", default=1000, action='store', help=" min size of cell in pixels")
    parser.add_argument("-ot","--overlap_threshold", type=float, dest="overlap_threshold", default=0.3, action='store', help="min percent overlap in decimal (0-1) for merge of 2d to 3d")
    parser.add_argument("-mt","--model_type", type=str, dest="model_type", default='nuclei', action='store', help="nuclei or cytoplasm")
    parser.add_argument("-g","--gpu", type=bool, dest="gpu", default=False, action='store', help="use the gpu?")
    parser.add_argument("-bs","--batch_size", type=int, dest="batch_size", default=8, action='store', help="cellpose input for breaking up image to batches")
    parser.add_argument("-d","--diameter", type=float, dest="diameter", default=90.0, action='store', help="diameter of cell in pixels")
    parser.add_argument("-c","--channels", type=list, dest="channels", default=[0,0], action='store', help="[0,0] for greyscale (default)")
    parser.add_argument("-ft","--flow_threshold", type=float, dest="flow_threshold", default=1.0, action='store', help="cellpose flow threshold")
    parser.add_argument("-ct","--cellprob_threshold", type=float, dest="cellprob_threshold", default=0.0, action='store', help="cllpose cell probability threshold")
    parser.add_argument("-pt","--pixel_thresh", type=int, dest="pixel_thresh", default=10**4.2, action='store', help="min area of cell in pixels")
    parser.add_argument("-dt","--distance_thresh", type=float, dest="distance_thresh", default=10, action='store', help="max distance from cell for voronoi")
    parser.add_argument("-zt","--z_thresh", type=int, dest="z_thresh", default=5, action='store', help="min z planes for keeping a cell")
    parser.add_argument("-o","--outpath", type=str, dest="outpath", default='infer', action='store', help="path to dataset or folder where mask acq will be saved")
    parser.add_argument("-f","--fresh", type=bool, dest="fresh", default=False, action='store', help="Overwrite previous masks?")
    args = parser.parse_args()
    
class ImageSegmentation(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 verbose=False):
#                  min_size=1000,
#                  overlap_threshold=0.3,
#                  model_type="nuclei",
#                  gpu=False,
#                  batch_size=8,
#                  diameter=90.0,
#                  channels = [0,0],
#                  flow_threshold=1,
#                  cellprob_threshold=0,
#                  pixel_thresh=10**4.2,
#                  z_thresh=5,
#                  distance_thresh=10,
#                  outpath='infer'):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.verbose = verbose
    
        self.pos_metadata = pos_metadata.sort_values(by='Zindex')
        self.verbose = verbose
        self.position = self.pos_metadata.Position.unique()[0]
        self.generate_stk()
        self.nZ = self.nuclear_stack.shape[2]
        self.nuclear_images = self.stack_to_images(self.nuclear_stack)
        self.min_size = min_size
        self.overlap_threshold = overlap_threshold
        self.pixel_thresh = pixel_thresh
        self.z_thresh = z_thresh
        self.distance_thresh = distance_thresh
        if outpath == 'infer':
            self.outpath = '/'+''.join([i+'/' for i in self.pos_metadata.filename.iloc[0].split('/')[1:-3]])
        else:
            self.outpath = outpath
            
        cellpose_inputs = {}
        cellpose_inputs['model_type'] = model_type
        cellpose_inputs['gpu'] = gpu
        cellpose_inputs['batch_size'] = batch_size
        cellpose_inputs['diameter'] = diameter
        cellpose_inputs['channels'] = channels
        cellpose_inputs['flow_threshold'] = flow_threshold
        cellpose_inputs['cellprob_threshold'] = cellprob_threshold
        self.cellpose_inputs = cellpose_inputs
       
    def generate_stk(self):
        stk = np.empty([2048,2048,len(self.pos_metadata)])
        for img_idx, fname in enumerate(self.pos_metadata.filename):
            # Weird print style to print on same line
            if self.verbose:
                sys.stdout.write("\r"+'opening '+os.path.split(fname)[-1])
                sys.stdout.flush()
            stk[:,:,img_idx]=io.imread(os.path.join(fname))
        self.nuclear_stack = stk
        
    def initalize_cellpose(self):
        self.model = models.Cellpose(model_type=self.cellpose_inputs['model_type'],
                                     gpu=self.cellpose_inputs['gpu'],
                                     batch_size=self.cellpose_inputs['batch_size'],
                                     verbose=self.verbose)
    def segment(self):
        self.raw_mask_images,flows,styles,diams = self.model.eval(self.nuclear_images,
                                              diameter=self.cellpose_inputs['diameter'],
                                              channels=self.cellpose_inputs['channels'],
                                              flow_threshold=self.cellpose_inputs['flow_threshold'],
                                              cellprob_threshold=self.cellpose_inputs['cellprob_threshold'])
        self.mask_images = self.raw_mask_images
        self.mask_stack = self.images_to_stack(self.mask_images)
    
    def stack_to_images(self,stack):
        return [stack[:,:,z] for z in range(self.nZ)]

    def images_to_stack(self,images):
        return np.stack(images,axis=2)
    
    def merge_labels_overlap(self,order):
        Input = self.mask_images
        Output = [np.zeros([2048,2048]) for i in range(self.nZ)]
        used_labels = 0
        if order == 'f':
            if self.verbose:
                iterable = tqdm(range(self.nZ),total=self.nZ,desc='forward merge')
            else:
                iterable = range(self.nZ)
            start = 0
            step = 1
        elif order == 'r':
            if self.verbose:
                iterable = tqdm(reversed(range(self.nZ)),total=self.nZ,desc='reverse merge')
            else:
                iterable = reversed(range(self.nZ))
            start = self.nZ-1
            step = -1
        for z in iterable:
            input_mask = Input[z]
            new_mask = np.zeros_like(input_mask)
            input_labels = np.unique(input_mask.ravel())
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
            iterable = tqdm(np.unique(mask_stack[mask_stack>0].ravel()),desc='filter')
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
        inverted_binary_mask_stk = self.mask_stack==0
        distance_mask_stk = dte(inverted_binary_mask_stk,sampling=[0.1,0.1,0.4])
        max_mask_stk = distance_mask_stk<self.distance_thresh
        labels = morphology.watershed(image=distance_mask_stk, markers=self.mask_stack,mask=max_mask_stk)
        self.voronoi_stack = labels
        self.voronoi_images = self.stack_to_images(labels)
        
    def generate_cell_metadata(self,stack):
        metadata = []
        for region in regionprops(stack):
            cell_id = str('cell_'+str(region.label)+'_'+str(self.position))
            x,y,z = np.array(region.centroid).astype(int)
            area = region.area
            metadata.append(pd.DataFrame([cell_id,x,y,z,area,pos],index=['cell_id','x_pixel','y_pixel','z_index','area','pos']).T)
        metadata = pd.concat(metadata,ignore_index=True)
        self.cell_metadata = metadata
        
    def run(self):
        self.initalize_cellpose()
        self.segment()
        self.merge_labels_overlap('f')
        self.merge_labels_overlap('r')
        self.filter_labels()
        self.voronoi()
        #self.generate_cell_metadata(self.voronoi_stack)
        
    def save_mask(self,acq,mask_stack):
        acq_path = os.path.join(self.outpath,acq)
        if not os.path.exists(acq_path):
            raise(NameError(acq_path,'Does not exist'))
        pos_path = os.path.join(acq_path,self.position)
        if not os.path.exists(pos_path):
            os.mkdir(pos_path)
        acq_metadata = self.pos_metadata.copy()
        filename = [os.path.join(self.position,self.position+'_'+acq+'_z_'+str(row.Zindex)+'.tif') for i,row in acq_metadata.iterrows()]
        root_pth = [os.path.join(acq_path,fname) for fname in filename]
        acq_metadata.root_pth = root_pth
        acq_metadata.filename = filename
        acq_metadata.acq = acq
        acq_metadata.XY = [str(i)[2:-1] for i in acq_metadata.XY]
        acq_metadata.XYbeforeTransform = [str(i)[2:-1] for i in acq_metadata.XYbeforeTransform]
        for i,img in enumerate(self.stack_to_images(mask_stack)):
            tifffile.imsave(root_pth[i], img.astype('uint16'))
        return acq_metadata.drop(columns=['root_pth']),acq_path
    
def create_acq(acq,md):
    acq_path = os.path.join(md.base_pth,acq)
    acq_metadata_path = os.path.join(acq_path,'Metadata.txt')
    if not os.path.exists(acq_path):
        os.mkdir(acq_path)
        acq_metadata = pd.DataFrame(columns = [i for i in md.image_table.columns if i !='filename'])
        acq_metadata.to_csv(acq_metadata_path,sep='\t',index=False)
    else:
        print(acq_path,'Already exists')
        acq_metadata = pd.read_csv(acq_metadata_path,sep='\t')
    return acq_metadata,acq_metadata_path

def segmentation_wrapper(pos_meta,
                         verbose=False,
                         min_size=1000,
                         overlap_threshold=0.3,
                         model_type="nuclei",
                         gpu=False,
                         batch_size=8,
                         diameter=90.0,
                         channels = [0,0],
                         flow_threshold=1,
                         cellprob_threshold=0,
                         pixel_thresh=10**4.2,
                         z_thresh=5,
                         distance_thresh=10,
                         outpath=None):
    segmentation_class = ImageSegmentation(pos_meta,
                                           verbose=verbose,
                                           min_size=min_size,
                                           overlap_threshold=overlap_threshold,
                                           model_type=model_type,
                                           gpu=gpu,
                                           batch_size=batch_size,
                                           diameter=diameter,
                                           channels = channels,
                                           flow_threshold=flow_threshold,
                                           cellprob_threshold=cellprob_threshold,
                                           pixel_thresh=pixel_thresh,
                                           z_thresh=z_thresh,
                                           distance_thresh=distance_thresh,
                                           outpath=outpath)
    segmentation_class.run()
    nuclei_metadata,nuclei_metadata_path = segmentation_class.save_mask('nuclei_mask',segmentation_class.mask_stack)
    voronoi_metadata,voronoi_metadata_path = segmentation_class.save_mask('voronoi_mask',segmentation_class.voronoi_stack)
    return {'nuclei_metadata':nuclei_metadata,'voronoi_metadata':voronoi_metadata}
    
if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['GOTO_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    md_path = args.md_path
    ncpu = args.ncpu
    nuclear_acq = args.nuclear_acq
    verbose=args.verbose
    min_size=args.min_size
    overlap_threshold=args.overlap_threshold
    model_type=args.model_type
    gpu=args.gpu
    batch_size=args.batch_size
    diameter=args.diameter
    channels = args.channels
    flow_threshold=args.flow_threshold
    cellprob_threshold=args.cellprob_threshold
    pixel_thresh=args.pixel_thresh
    z_thresh=args.z_thresh
    distance_thresh=args.distance_thresh
    outpath=args.outpath
    fresh = args.fresh
    print(args)
    
    md = Metadata(md_path)
    if 'infer' == nuclear_acq:
        acq = [i for i in md.acqnames if 'nucstain' in i][0]
    else:
        acq = nuclear_acq
    nuclei_metadata_master,nuclei_metadata_master_path = create_acq('nuclei_mask',md)
    voronoi_metadata_master,voronoi_metadata_master_path = create_acq('voronoi_mask',md)
    finished_poses = set(list(nuclei_metadata_master.Position.unique())).intersection(list(voronoi_metadata_master.Position.unique()))
    Input = []
    for pos in md.image_table[(md.image_table.acq==acq)].Position.unique():
        if fresh:
            pos_meta = md.image_table[(md.image_table.Position==pos)&(md.image_table.acq==acq)&(md.image_table.Channel=='DeepBlue')].copy()
            Input.append(pos_meta)
        else:
            if pos not in finished_poses:
                pos_meta = md.image_table[(md.image_table.Position==pos)&(md.image_table.acq==acq)&(md.image_table.Channel=='DeepBlue')].copy()
                Input.append(pos_meta)
        
    pfunc = partial(segmentation_wrapper,
                    verbose=verbose,
                    min_size=min_size,
                    overlap_threshold=overlap_threshold,
                    model_type=model_type,
                    gpu=gpu,
                    batch_size=batch_size,
                    diameter=diameter,
                    channels = channels,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    pixel_thresh=pixel_thresh,
                    z_thresh=z_thresh,
                    distance_thresh=distance_thresh,
                    outpath=outpath)
    if ncpu==1:
        for pos in md.image_table[(md.image_table.acq==acq)].Position.unique():  
            pos_meta = md.image_table[(md.image_table.Position==pos)&(md.image_table.acq==acq)&(md.image_table.Channel=='DeepBlue')].copy()
            out_dict = pfunc(pos_meta)
            nuclei_metadata_master = pd.concat([nuclei_metadata_master,out_dict['nuclei_metadata']])
            voronoi_metadata_master = pd.concat([voronoi_metadata_master,out_dict['voronoi_metadata']])
            nuclei_metadata_master.drop_duplicates(subset='filename').to_csv(nucl+ei_metadata_master_path,sep='\t',index=False)
            voronoi_metadata_master.drop_duplicates(subset='filename').to_csv(voronoi_metadata_master_path,sep='\t',index=False)
    else:
        Input = []
        for pos in md.image_table[(md.image_table.acq==acq)].Position.unique():
            pos_meta = md.image_table[(md.image_table.Position==pos)&(md.image_table.acq==acq)&(md.image_table.Channel=='DeepBlue')].copy()
            Input.append(pos_meta)
        with multiprocessing.Pool(ncpu) as ppool:
            sys.stdout.flush()
            for out_dict in tqdm(ppool.imap(pfunc, Input),total=len(Input)):
                nuclei_metadata_master = pd.concat([nuclei_metadata_master,out_dict['nuclei_metadata']])
                voronoi_metadata_master = pd.concat([voronoi_metadata_master,out_dict['voronoi_metadata']])
                nuclei_metadata_master.drop_duplicates(subset='filename').to_csv(nuclei_metadata_master_path,sep='\t',index=False)
                voronoi_metadata_master.drop_duplicates(subset='filename').to_csv(voronoi_metadata_master_path,sep='\t',index=False)
            ppool.close()
            sys.stdout.flush()