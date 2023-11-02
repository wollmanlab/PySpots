# Config for analysis of SeqFish data in Wollman lab (Jan 2018)
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

# Basic parameters of imaging

depricated_bitmap = [('RS0095_cy5', 'hybe2', 'FarRed'), 
                     ('RS0109_cy5', 'hybe4', 'FarRed'),
                     ('RS0175_cy5', 'hybe6', 'FarRed'), 
                     ('RS0237_cy5', 'hybe1', 'FarRed'),
                     ('RS0307_cy5', 'hybe3', 'FarRed'), 
                     ('RS0332_cy5', 'hybe5', 'FarRed'),
                     ('RS0384_atto565', 'hybe5', 'Orange'), 
                     ('RS0406_atto565', 'hybe6', 'Orange'),
                     ('RS0451_atto565', 'hybe4', 'Orange'), 
                     ('RS0468_atto565', 'hybe3', 'Orange'),
                     ('RS0548_atto565', 'hybe2', 'Orange'), 
                     ('RS64.0_atto565', 'hybe1', 'Orange'),
                     ('RSN9927.0_cy5', 'hybe8', 'FarRed'), 
                     ('RSN2336.0_cy5', 'hybe7', 'FarRed'), 
                     ('RSN1807.0_cy5', 'hybe9', 'FarRed'), 
                     ('RSN4287.0_atto565', 'hybe7', 'Orange'), 
                     ('RSN1252.0_atto565', 'hybe9', 'Orange'), 
                     ('RSN9535.0_atto565', 'hybe8', 'Orange')]
          
bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'), 
          ('RS0109_cy5', 'hybe3', 'FarRed'),
          ('RS0175_cy5', 'hybe5', 'FarRed'), 
          ('RS0237_cy5', 'hybe6', 'FarRed'),
          ('RS0307_cy5', 'hybe2', 'FarRed'), 
          ('RS0332_cy5', 'hybe4', 'FarRed'),
          ('RS0384_atto565', 'hybe4', 'Orange'), 
          ('RS0406_atto565', 'hybe5', 'Orange'),
          ('RS0451_atto565', 'hybe3', 'Orange'), 
          ('RS0468_atto565', 'hybe2', 'Orange'),
          ('RS0548_atto565', 'hybe1', 'Orange'), 
          ('RS64.0_atto565', 'hybe6', 'Orange'),
          ('RSN9927.0_cy5', 'hybe8', 'FarRed'), 
          ('RSN2336.0_cy5', 'hybe7', 'FarRed'), 
          ('RSN1807.0_cy5', 'hybe9', 'FarRed'), 
          ('RSN4287.0_atto565', 'hybe7', 'Orange'), 
          ('RSN1252.0_atto565', 'hybe9', 'Orange'), 
          ('RSN9535.0_atto565', 'hybe8', 'Orange')]

nbits = len(bitmap)

codebook_pth = '/bigstore/GeneralStorage/Zach/MERFISH/Cornea/Inflammation.txt'
def load_codebook(fname):
    barcodes = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            bc = map(int, line.strip().split(','))
            barcodes.append(list(bc))
    return np.array(barcodes)

# cwords = load_codebook('/home/rfor10/repos/seqfish_design/MHD4_24bit_472cwords.csv')
# Import the codebook for genes in the experiment
base_pth = '/home/zach/PythonRepos/PySpots/hybescope_config/'
possible_cwords = load_codebook(os.path.join(base_pth,'MHD4_18bit_187cwords.csv'))

codebook = pd.read_csv(codebook_pth,skiprows=3)
codebook.columns = [i.split(' ')[-1] for i in codebook.columns]
codebook['barcode'] = [str(i).zfill(nbits) for i in codebook['barcode']]
blank_cwords = []
blank_names = []
for i in range(possible_cwords.shape[0]):
    barcode = ''.join([str(b) for b in possible_cwords[i,:]])
    if not barcode in list(codebook['barcode']):
        blank_cwords.append(possible_cwords[i,:])
        blank_names.append('blank'+str(len(blank_names)))
blank_cwords = np.stack(blank_cwords)
true_cwords = np.array([np.array([int(i) for i in codebook['barcode'].iloc[b]]) for b in range(len(codebook))])
cwords = np.concatenate([true_cwords,blank_cwords])

gids = list(codebook['name'])
bids = blank_names
aids = gids+bids
gene_codeword_vectors = true_cwords
blank_codeword_vectors = blank_cwords
all_codeword_vectors = cwords
norm_gene_codeword_vectors = normalize(gene_codeword_vectors)
norm_blank_codeword_vectors = normalize(blank_codeword_vectors)
norm_all_codeword_vectors = normalize(all_codeword_vectors)

parameters = {}
parameters['dtype_rel_min']=0
parameters['dtype_rel_max']=100
parameters['dtype']='uint16'
parameters['background_kernel']=9
parameters['blur_kernel']=1
parameters['background_method']='median'
parameters['blur_method']='gaussian'
parameters['hotpixel_kernel_size']=3
parameters['normalization_rel_min']=50
parameters['normalization_rel_max']=95
parameters['deconvolution_niterations']=10
parameters['deconvolution_batches']=10
parameters['deconvolution_gpu']=False
parameters['projection_zstart']=-1
parameters['projection_k']=1
parameters['projection_zskip']=2 
parameters['projection_zend']=-1
parameters['projection_function']='mean'
parameters['verbose']=False
parameters['ncpu']=10
parameters['normalization_max']=1000
parameters['normalization_rel_max']=99
parameters['normalization_rel_min']=50
parameters['registration_threshold']=2
parameters['upsamp_factor']=5
parameters['dbscan_eps']=3
parameters['dbscan_min_samples']=20
parameters['max_dist']=200
parameters['match_threshold']=0.65
parameters['ref_hybe']='hybe1'
parameters['hybedata']='hybedata'
parameters['fishdata']='fishdata'
parameters['registration_channel']='DeepBlue'
parameters['daemon_path']='/scratch/daemon/'
parameters['utilities_path']='/scratch/utilities/'
parameters['floor']=True
parameters['two_dimensional']=False
parameters['match_thresh'] = -2
parameters['fpr_thresh'] = 0.4
parameters['nucstain_channel'] = 'DeepBlue'
parameters['nucstain_acq'] = 'nucstain'
parameters['registration_method'] = 'beads'
parameters['segment_projection_function'] = 'mean'
parameters['segment_min_size'] = 1000
parameters['segment_overlap_threshold'] = 0.3
parameters['segment_pixel_thresh'] = 10**3#10**4
parameters['segment_z_thresh'] = 0#5
parameters['segment_distance_thresh'] = 10
parameters['segment_model_type']="nuclei"
parameters['segment_gpu'] = False
parameters['segment_batch_size'] = 8
parameters['segment_diameter'] = 90.0
parameters['segment_channels'] = [0,0]
parameters['segment_flow_threshold'] = 1
parameters['segment_cellprob_threshold'] = 0
parameters['segment_downsample'] = 0.25
parameters['segment_two_dimensional'] = True#False
parameters['segment_overwrite'] = False
parameters['segment_singular_zindex'] = -1
parameters['segment_nuclear_blur'] = 300
parameters['segment_pixel_size'] = 0.103
parameters['segment_z_step_size'] = 0.4

hotpixel_loc = pickle.load(open('/scratch/hotpixels.pkl','rb'))
hotpixel_X = hotpixel_loc[0]
hotpixel_Y = hotpixel_loc[1]