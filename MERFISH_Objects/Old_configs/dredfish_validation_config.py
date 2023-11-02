# Update Jan 2020
# Appears that the hybe solutions may have changed when Zach prepared them
# in late 2019 (hybe6 and hybe7 maybe swapped) so this is config to test that
# Config for analysis of SeqFish data in Wollman lab (Jan 2018)
# Necessary files are assumed to be in the current directory when this config is imported.
# Currently designed for Python 3.5

################################ Probe/Codebook Related Config ##########################
# This bitmap is tyically constant for most experiments
# Might need to change this if experimental conditions are different
import numpy
import numpy as np
import dill as pickle
import pandas as pd
import os
from scipy.spatial import distance_matrix
from collections import OrderedDict
from sklearn.preprocessing import normalize
from MERFISH_Objects.hybescope_config.microscope_config import *

# Basic parameters of imaging
# bitmap = [('RS0109_cy5', 'hybe1', 'FarRed'),
#           ('RS0175_cy5', 'hybe2', 'FarRed'),
#           ('RS0237_cy5', 'hybe3', 'FarRed'),
#           ('RS0307_cy5', 'hybe4', 'FarRed'),
#           ('RS0332_cy5', 'hybe5', 'FarRed'),
#           ('RS0384_atto565', 'hybe6', 'FarRed'),
#           ('RS0406_atto565', 'hybe7', 'FarRed'),
#           ('RS0451_atto565', 'hybe8', 'FarRed'),
#           ('RS0468_atto565', 'hybe9', 'FarRed'),
#           ('RS0548_atto565', 'hybe10', 'FarRed'),
#           ('RS64.0_atto565', 'hybe11', 'FarRed'),
#           ('RS156.0_alexa488', 'hybe12', 'FarRed'),
#           ('RS278.0_alexa488', 'hybe13', 'FarRed'),
#           ('RS313.0_alexa488', 'hybe14', 'FarRed'),
#           ('RS643.0_alexa488', 'hybe15', 'FarRed'),
#           ('RS740.0_alexa488', 'hybe16', 'FarRed'),
#           ('RS810.0_alexa488', 'hybe17', 'FarRed'),
#           ('RSN9927.0_cy5', 'hybe18', 'FarRed'),
#           ('RSN2336.0_cy5', 'hybe19', 'FarRed'),
#           ('RSN1807.0_cy5', 'hybe20', 'FarRed'),
#           ('RSN4287.0_atto565', 'hybe21', 'FarRed'),
#           ('RSN1252.0_atto565', 'hybe22', 'FarRed'),
#           ('RSN9535.0_atto565', 'hybe23', 'FarRed'),
#           ('RS0095_cy5', 'hybe24', 'FarRed')]
bitmap = [('516094.0', 'hybe1', 'FarRed','RS516094_RS0109'),
          ('336946.0', 'hybe2', 'FarRed','RS336946_RS0175'),
          ('617691.0', 'hybe3', 'FarRed','RS617691_RS0237'),
          ('1038995.0', 'hybe4', 'FarRed','RS1038995_RS0307'),
          ('106372.0', 'hybe5', 'FarRed','RS106372_RS0332'),
          ('343998.0', 'hybe6', 'FarRed','RS343998_RS0384'),
          ('5261.0', 'hybe7', 'FarRed','RS5261_RS0406'),
          ('71182.0', 'hybe8', 'FarRed','RS71182_RS0451'),
          ('50432.0', 'hybe9', 'FarRed','RS50432_RS0468'),
          ('74925.0', 'hybe10', 'FarRed','RS74925_RS0548'),
          ('362116.0', 'hybe11', 'FarRed','RS362116_RS64.0'),
          ('687380.0', 'hybe12', 'FarRed','RS687380_RS156.0'),
          ('154669.0', 'hybe13', 'FarRed','RS154669_RS278.0'),
          ('860204.0', 'hybe14', 'FarRed','RS860204_RS313.0'),
          ('8916.0', 'hybe15', 'FarRed','RS8916_RS643.0'),
          ('695682.0', 'hybe16', 'FarRed','RS695682_RS740.0'),
          ('245993.0', 'hybe17', 'FarRed','RS245993_RS810.0'),
          ('277436.0', 'hybe18', 'FarRed','RS277436_RS0095')]
bitmap = [i[0:3] for i in bitmap]

nbits = len(bitmap)

# config_options
codebook_pth = '/bigstore/binfo/Codebooks/dredFISH_Validation_Codebook.csv'
base_pth = '/home/zach/PythonRepos/PySpots/hybescope_config/'
         
# Import the codebook for genes in the experiment
codewords = pd.read_csv(codebook_pth,index_col=0)
codewords.columns = ['name', 'id', 'barcode']
bcs = []
for bc in codewords.barcode:
    bc = str(bc)
    if len(bc)<nbits:
        bc = '0'*(nbits-len(bc))+bc
    bcs.append(bc)
codewords.barcode = bcs

f = open('/bigstore/binfo/mouse/Brain/DRedFISH/Validation_Final_oligos.fasta', 'r')
s = f.read()
f.close()
present = [i in s for i in codewords.id.values]
codewords = codewords[present]

def load_codebook(fname):
    barcodes = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            bc = map(int, line.strip().split(','))
            barcodes.append(list(bc))
    return np.array(barcodes)

# Import the codebook for genes in the experiment
possible_cwords = load_codebook(os.path.join(base_pth,'MHD4_18bit_187cwords.csv'))

codebook = pd.read_csv(codebook_pth,index_col=0)
codebook.columns = [i.split(' ')[-1] for i in codebook.columns]
codebook['barcode'] = [str(i).zfill(nbits) for i in codebook['barcode']]
blank_cwords = []
blank_names = []

true_cwords = np.array([np.array([int(i) for i in codebook['barcode'].iloc[b]]) for b in range(len(codebook))])
total_cwords = np.zeros_like(possible_cwords)
total_cwords[0:true_cwords.shape[0],:] = true_cwords
template = np.zeros(true_cwords.shape[1]).astype(int)
ncwords = true_cwords.shape[0]
template[0:4] = 1
while ncwords<possible_cwords.shape[0]:
    hamming_distance = np.abs(true_cwords-template).sum(axis=1).min()
    if hamming_distance>=4:
        total_cwords[ncwords,:] = template
        ncwords+=1
    np.random.shuffle(template)
blank_cwords = total_cwords[true_cwords.shape[0]:,:]
cwords = np.concatenate([true_cwords,blank_cwords])
blank_names = ['blank'+str(i) for i in range(blank_cwords.shape[0])]
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
parameters['background_kernel']=7
parameters['blur_kernel']=0.9
parameters['background_method']='median'
parameters['blur_method']='gaussian'
parameters['hotpixel_kernel_size']=3
parameters['normalization_rel_min']=50
parameters['normalization_rel_max']=95
parameters['deconvolution_niterations']=0#10
parameters['deconvolution_batches']=10
parameters['deconvolution_gpu']=False
parameters['projection_zstart']=-1
parameters['projection_k']=0
parameters['projection_zskip']=1 
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
parameters['2D']=True
parameters['nucstain_acq'] = 'nucstain'
parameters['two_dimensional'] = True
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
parameters['registration_method'] = 'image'
parameters['match_thresh'] = -2
parameters['fpr_thresh'] = 0.2
parameters['nucstain_channel'] = 'DeepBlue'

hotpixel_loc = pickle.load(open('/scratch/hotpixels.pkl','rb'))
# Not certain about hotpixel x vs y
hotpixel_X = hotpixel_loc[0]
hotpixel_Y = hotpixel_loc[1]