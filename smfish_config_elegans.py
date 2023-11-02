# Config for analysis of MERFISH data in Wollman lab (Jan 2018)
# Necessary files are assumed to be in the current directory when this config is imported.
# Currently designed for Python 3.5

################################ Probe/Codebook Related Config ##########################
# This bitmap is tyically constant for most experiments
# Might need to change this if experimental conditions are different
import numpy as np
import pickle
import pandas as pd
import os
from scipy.spatial import distance_matrix
from collections import OrderedDict
from sklearn.preprocessing import normalize
from MERFISH_Objects.hybescope_config.microscope_config import *

""" Order in which the barcode was imaged and with which color"""
bitmap = [('PolyT_RS458122', 'hybe1', 'FarRed'),
            ('elt-2_RS156.0', 'hybe2', 'FarRed'),
            ('cdc-25.1_RS0451', 'hybe3', 'FarRed'),
            ('pha-4_RSN9927.0', 'hybe4', 'FarRed')]


nbits = len(bitmap)

""" MERFISH Code Parameters"""
parameters = {}
""" General """
parameters['daemon_path']= '/greendata/GeneralStorage/daemon' #'/scratch/daemon/' # Where should the Daemons Look
parameters['utilities_path']= '/greendata/GeneralStorage/utilities'#'/scratch/utilities/' # Where to save temporary files
parameters['fishdata']='fishdata_3D' #Directory Name for Processed Data >Bigstore>Images[Year]>User.Project>Dataset>fishdata
parameters['verbose']=False # If you want print statements (Mostly for diagnostics)
parameters['two_dimensional']=False #Work in 2 or 3 dimensions
parameters['pixel_size'] = 0.083 #1.5 => 0.083 #1=> 0.123# size of pixel in um
parameters['z_step_size'] = 0.4 # size of step in Z um
""" Dataset """
parameters['hotpixel_kernel_size']=3 # Size of window to calculate hot pixel
parameters['std_thresh']=3 # Threshhold in std for calling hot pixels
parameters['n_acqs']=5 # number of acquisitions to use to calculate hot pixels
""" Position """

""" Hybe """

""" Registration """
parameters['registration_threshold']=2 # Max average distance in pixels before error
parameters['upsamp_factor']=5 # How much to upsample to call subpixel center
parameters['dbscan_eps']=3 
parameters['dbscan_min_samples']=20 # How many beads you need
parameters['max_dist']=int(15/parameters['pixel_size'])#15 um # Max distance a bead can be to be paired
parameters['match_threshold']=0.65 # Peak Calling threshold for beads
parameters['ref_hybe']='hybe1' # Name of Reference Hybe
parameters['registration_channel']='DeepBlue' # What channel has your fiduciary markers
parameters['registration_method'] = 'beads' # What method of registration to use (bead or image)
parameters['registration_image_blur_kernel'] = (0.1/parameters['pixel_size']) #100 nm # What size kernel to blur image with
parameters['registration_image_background_kernel'] = (0.5/parameters['pixel_size']) # 500nm # What size kernel to calculte background
parameters['subpixel_method'] = 'max'
parameters['registration_overwrite'] = False
""" Stack """

""" Image """
parameters['projection_zstart']=-1 # Which Z index to start (-1 means first)
parameters['projection_k']=0#1 # How many Z above and below to include (1 means 1 above and 1 below)
parameters['projection_zskip']=1#2 # How many Z indexes to skip
parameters['projection_zend']=-1 # Which Z index to stop (-1 means last)
parameters['projection_function']='max' # Which method to use to project (typically max or mean)
parameters['dtype_rel_min']=0 # when converting dtypes this amount in percentile will be set to 0
parameters['dtype_rel_max']=100 # when converting dtypes this amount in percentile will be set to max
parameters['dtype']='uint16' # dtype to save in
parameters['background_kernel']=(0.6/parameters['pixel_size']) #200 nm size of kernel for background in pixels (ideally just larger than spots)
parameters['blur_kernel']=(0.1/parameters['pixel_size'])#50nm amount to blur your image to smooth out noise
parameters['background_method']='gaussian' # method to calculate background
parameters['blur_method']='gaussian' # method to smooth image
parameters['deconvolution_niterations']=0 # How many rounds of deconvolution to perform
parameters['deconvolution_batches']=10 # how many batches to break up the computation into
parameters['deconvolution_gpu']=False # do you want to use the gpu
parameters['gain'] = 10 # gain for saving Images to use more dynamic range 
parameters['spot_diameter'] = 5 # 250 nm
parameters['spot_minmass'] = 15#9 # not based on size?
parameters['spot_separation'] = 3 # 100 nm
parameters['image_call_spots'] = False
parameters['image_overwrite'] = False
""" Segment """
parameters['nucstain_channel'] = 'DeepBlue' # Which Channel is your nuclear stain in
parameters['nucstain_acq'] = 'hybe1' # Which acquision is your nuclear signal in
parameters['segment_projection_function'] = 'mean' # method for projecting in Z
parameters['segment_min_size'] = (100/parameters['pixel_size'])#100 um2 # min cell size in pixels to keep for 2D
parameters['segment_overlap_threshold'] = 0.3 # % of overlap to merge cells in Z
parameters['segment_pixel_thresh'] = (100/parameters['pixel_size'])#100 um2 # min cell size in pixels to keep for 3D
parameters['segment_z_thresh'] = 0#5 # How many Z's a cell has to be in to keep
parameters['segment_distance_thresh'] = 10 # distance to dialate cell in um
parameters['segment_model_type']="nuclei" # cellpose model type
parameters['segment_gpu'] = False # use gpu?
parameters['segment_batch_size'] = 8 # how many batches to break up calculation
parameters['segment_diameter'] = (5/parameters['pixel_size']) # size of cell in um2
parameters['segment_channels'] = [0,0] # grey scale for cellpose
parameters['segment_flow_threshold'] = 1 # cellpose parameters
parameters['segment_cellprob_threshold'] = 0 # cellpose parameters
parameters['segment_downsample'] = 1 # amount to downsample images
parameters['segment_two_dimensional'] = True#False # perform in 2D or 3D
parameters['segment_overwrite'] = True # Overwrite previous segmentation?
parameters['segment_nuclear_blur'] = (25/parameters['pixel_size']) # 25 um2 sigma in pixels for background
parameters['segment_z_step_size'] = 0.4
parameters['segment_pixel_size'] = parameters['pixel_size']
parameters['segment_ncpu'] = 30
""" Classify """
parameters['classification_method'] = 'spot'
parameters['match_thresh'] = -2 # how many mismatched bits to be called a barcode
parameters['fpr_thresh'] = 0.4 # euclidean distance from barcodes to be called
parameters['classification_overwrite'] = False
parameters['spot_percentile'] = 99.9
parameters['spot_max_distance'] = 0.1
parameters['spot_diameter'] = 5
parameters['spot_minmass'] = 4
parameters['spot_separation'] = 3
parameters['classify_iterations'] = 1
#parameters['logistic_columns'] = ['mass', 'size', 'ecc','intensity' 'raw_mass', 'ep', 'cword_distance', 'correct_bits', 'false_positives', 'false_negatives', 'signal', 'noise', 'signal-noise', 'X']
parameters['logistic_columns'] = ['raw_mass', 'ep','intensity', 'signal', 'noise', 'signal-noise','X']
spot_parameters = {}
spot_parameters['default'] = {'spot_max_distance':3,
                                       'spot_minmass':5,
                                       'spot_diameter':5,
                                          'spot_separation':3}
spot_parameters['WT_2022Sep16'] = {'spot_max_distance':0.1,
                                       'spot_minmass':4,
                                       'spot_diameter':5,
                                          'spot_separation':3}
# spot_parameters['sham1_3_2022Jan03'] = {'spot_max_distance':3,
#                                        'spot_minmass':12,
#                                        'spot_diameter':5,
#                                           'spot_separation':3}
# spot_parameters['Sham_2_4_2022Jan21'] = {'spot_max_distance':3,
#                                        'spot_minmass':17,
#                                        'spot_diameter':5,
#                                           'spot_separation':3}
# spot_parameters['Sham_3_4_2022Jan25'] = {'spot_max_distance':3,
#                                        'spot_minmass':15,
#                                        'spot_diameter':5,
#                                           'spot_separation':3}
# spot_parameters['TBI-1-1_2021Oct20'] = {'spot_max_distance':3,
#                                        'spot_minmass':15,
#                                        'spot_diameter':5,
#                                           'spot_separation':3}
# spot_parameters['TBI_3_1_2022Jan15'] = {'spot_max_distance':3,
#                                        'spot_minmass':19,
#                                        'spot_diameter':5,
#                                           'spot_separation':3}
# spot_parameters['TBI4_1_2022Jan11'] = {'spot_max_distance':3,
#                                        'spot_minmass':15,
#                                        'spot_diameter':5,
#                                           'spot_separation':3}


camera_direction_dict = {'default':[-1,-1],
                        'sham1_3_2022Jan03':[-1,-1],    
                        'Sham_2_4_2022Jan21':[-1,-1],
                        'Sham_3_4_2022Jan25':[-1,-1],
                        'TBI-1-1_2021Oct20':[-1,-1],
                        'TBI_3_1_2022Jan15':[-1,-1],
                        'TBI4_1_2022Jan11':[-1,-1]}
xy_flip_dict = {'default':False,
                'sham1_3_2022Jan03':True,
                'Sham_2_4_2022Jan21':False,
                'Sham_3_4_2022Jan25':False,
                'TBI-1-1_2021Oct20':True,
                'TBI_3_1_2022Jan15':False,
                'TBI4_1_2022Jan11':True}

parameters['camera_direction_dict'] = camera_direction_dict
parameters['xy_flip_dict'] = xy_flip_dict
parameters['spot_parameters'] = spot_parameters