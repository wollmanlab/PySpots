# Update Jan 2020
# Appears that the hybe solutions may have changed when Zach prepared them
# in late 2019 (hybe6 and hybe7 maybe swapped) so this is config to test that
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
from hybescope_config.microscope_config import *
import dill as pickle

# # Basic parameters of imaging 
bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
          ('RS0109_cy5', 'hybe3', 'FarRed'),
          ('RS0175_cy5', 'hybe5', 'FarRed'),
          ('RS0237_cy5', 'hybe6', 'FarRed'), #Check 7
          ('RS0307_cy5', 'hybe2', 'FarRed'),
          ('RS0332_cy5', 'hybe4', 'FarRed'),
          ('RS0384_atto565', 'hybe4', 'Orange'),
          ('RS0406_atto565', 'hybe5', 'Orange'),
          ('RS0451_atto565', 'hybe3', 'Orange'),
          ('RS0468_atto565', 'hybe2', 'Orange'),
          ('RS0548_atto565', 'hybe1', 'Orange'),
          ('RS64.0_atto565', 'hybe6', 'Orange'),#Check 7
          ('RSN9927.0_cy5', 'hybe8', 'FarRed'),
          ('RSN2336.0_cy5', 'hybe7', 'FarRed'), #Check 6
          ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
          ('RSN4287.0_atto565', 'hybe7', 'Orange'), #Check 6
          ('RSN1252.0_atto565', 'hybe9', 'Orange'),
          ('RSN9535.0_atto565', 'hybe8', 'Orange')]


nbits = len(bitmap)

def load_codebook(fname):
    barcodes = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            bc = map(int, line.strip().split(','))
            barcodes.append(list(bc))
    return np.array(barcodes)

# config_options
codebook_pth = '/bigstore/GeneralStorage/Evan/NFKB_MERFISH/BMDM_NFkB_v2/Probes/Genes_Final_120919/design/codebook_120919.csv'
base_pth = '/home/zach/PythonRepos/PySpots/hybescope_config/'
possible_cwords = load_codebook(os.path.join(base_pth,'MHD4_18bit_187cwords.csv'))

# Import the codebook for genes in the experiment
codebook = pd.read_csv(codebook_pth,skiprows=3)
codebook.columns = [i.split(' ')[-1] for i in codebook.columns]
codebook['barcode'] = [str(i).zfill(nbits) for i in codebook['barcode']]
blank_cwords = []
blank_names = []
for i in range(possible_cwords.shape[0]):
    barcode = ''.join([str(b) for b in possible_cwords[i,:]])
    if not barcode in list(codebook['barcode']):
        blank_cwords.append(possible_cwords[i,:])
        blank_names.append('blank_'+str(len(blank_names)))
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
parameters['background_kernel']=15
parameters['blur_kernel']=0
parameters['background_method']='gaussian'
parameters['blur_method']='gaussian'
parameters['hotpixel_kernel_size']=3
parameters['normalization_rel_min']=50
parameters['normalization_rel_max']=90
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
parameters['match_thresh'] = -2
parameters['fpr_thresh'] = 0.2

hotpixel_loc = pickle.load(open('/scratch/hotpixels.pkl','rb'))
# Not certain about hotpixel x vs y
hotpixel_X = hotpixel_loc[0]
hotpixel_Y = hotpixel_loc[1]