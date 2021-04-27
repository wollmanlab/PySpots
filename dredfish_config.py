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
from hybescope_config.microscope_config import *

# Basic parameters of imaging
bitmap = [('RS0109_cy5', 'hybe1', 'FarRed'),
          ('RS0175_cy5', 'hybe2', 'FarRed'),
          ('RS0237_cy5', 'hybe3', 'FarRed'),
          ('RS0307_cy5', 'hybe4', 'FarRed'),
          ('RS0332_cy5', 'hybe5', 'FarRed'),
          ('RS0384_atto565', 'hybe6', 'FarRed'),
          ('RS0406_atto565', 'hybe7', 'FarRed'),
          ('RS0451_atto565', 'hybe8', 'FarRed'),
          ('RS0468_atto565', 'hybe9', 'FarRed'),
          ('RS0548_atto565', 'hybe10', 'FarRed'),
          ('RS64.0_atto565', 'hybe11', 'FarRed'),
          ('RS156.0_alexa488', 'hybe12', 'FarRed'),
          ('RS278.0_alexa488', 'hybe13', 'FarRed'),
          ('RS313.0_alexa488', 'hybe14', 'FarRed'),
          ('RS643.0_alexa488', 'hybe15', 'FarRed'),
          ('RS740.0_alexa488', 'hybe16', 'FarRed'),
          ('RS810.0_alexa488', 'hybe17', 'FarRed'),
          ('RSN9927.0_cy5', 'hybe18', 'FarRed'),
          ('RSN2336.0_cy5', 'hybe19', 'FarRed'),
          ('RSN1807.0_cy5', 'hybe20', 'FarRed'),
          ('RSN4287.0_atto565', 'hybe21', 'FarRed'),
          ('RSN1252.0_atto565', 'hybe22', 'FarRed'),
          ('RSN9535.0_atto565', 'hybe23', 'FarRed'),
          ('RS0095_cy5', 'hybe24', 'FarRed')]

# depricated_bitmap = [('RS0095_cy5', 'hybe2', 'FarRed'), ('RS0109_cy5', 'hybe4', 'FarRed'),
#           ('RS0175_cy5', 'hybe6', 'FarRed'), ('RS0237_cy5', 'hybe1', 'FarRed'),
#           ('RS0307_cy5', 'hybe3', 'FarRed'), ('RS0332_cy5', 'hybe5', 'FarRed'),
#           ('RS0384_atto565', 'hybe5', 'Orange'), ('RS0406_atto565', 'hybe6', 'Orange'),
#           ('RS0451_atto565', 'hybe4', 'Orange'), ('RS0468_atto565', 'hybe3', 'Orange'),
#           ('RS0548_atto565', 'hybe2', 'Orange'), ('RS64.0_atto565', 'hybe1', 'Orange'),
#           ('RSN9927.0_cy5', 'hybe8', 'FarRed'), ('RSN2336.0_cy5', 'hybe7', 'FarRed'), 
#           ('RSN1807.0_cy5', 'hybe9', 'FarRed'), ('RSN4287.0_atto565', 'hybe7', 'Orange'), 
#           ('RSN1252.0_atto565', 'hybe9', 'Orange'), ('RSN9535.0_atto565', 'hybe8', 'Orange')]
          
# bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'), ('RS0109_cy5', 'hybe3', 'FarRed'),
#           ('RS0175_cy5', 'hybe5', 'FarRed'), ('RS0237_cy5', 'hybe7', 'FarRed'),
#           ('RS0307_cy5', 'hybe2', 'FarRed'), ('RS0332_cy5', 'hybe4', 'FarRed'),
#           ('RS0384_atto565', 'hybe4', 'Orange'), ('RS0406_atto565', 'hybe5', 'Orange'),
#           ('RS0451_atto565', 'hybe3', 'Orange'), ('RS0468_atto565', 'hybe2', 'Orange'),
#           ('RS0548_atto565', 'hybe1', 'Orange'), ('RS64.0_atto565', 'hybe7', 'Orange'),
#           ('RS156.0_alexa488', 'hybe2', 'Green'), ('RS278.0_alexa488', 'hybe3','Green'),
#           ('RS313.0_alexa488', 'hybe4', 'Green'), ('RS643.0_alexa488', 'hybe6', 'Green'),
#           ('RS740.0_alexa488', 'hybe1', 'Green'), ('RS810.0_alexa488', 'hybe5', 'Green'),
#           ('RSN9927.0_cy5', 'hybe8', 'FarRed'), ('RSN2336.0_cy5', 'hybe6', 'FarRed'), 
#           ('RSN1807.0_cy5', 'hybe9', 'FarRed'), ('RSN4287.0_atto565', 'hybe6', 'Orange'), 
#           ('RSN1252.0_atto565', 'hybe9', 'Orange'), ('RSN9535.0_atto565', 'hybe8', 'Orange')
#          ]
# # The order of the sequences in the codebook was changed for this experiment during sequence design.
# # The oligos are still in the same hybe order, but the order of the tuples changed to group all the 
# # FarRed and Orange sequences consequtively.
# bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'), ('RS0109_cy5', 'hybe3', 'FarRed'),
#           ('RS0175_cy5', 'hybe5', 'FarRed'), ('RS0237_cy5', 'hybe7', 'FarRed'),
#           ('RS0307_cy5', 'hybe2', 'FarRed'), ('RS0332_cy5', 'hybe4', 'FarRed'),
#           ('RSN9927.0_cy5', 'hybe8', 'FarRed'), ('RSN2336.0_cy5', 'hybe6', 'FarRed'), 
#           ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
#           ('RS0384_atto565', 'hybe4', 'Orange'), ('RS0406_atto565', 'hybe5', 'Orange'),
#           ('RS0451_atto565', 'hybe3', 'Orange'), ('RS0468_atto565', 'hybe2', 'Orange'),
#           ('RS0548_atto565', 'hybe1', 'Orange'), ('RS64.0_atto565', 'hybe7', 'Orange'),
#           ('RSN4287.0_atto565', 'hybe6', 'Orange'), 
#           ('RSN1252.0_atto565', 'hybe9', 'Orange'), ('RSN9535.0_atto565', 'hybe8', 'Orange')]

nbits = len(bitmap)

# config_options
codebook_pth = '/bigstore/GeneralStorage/Rob/merfish/MERFISH_analysis-master/mouse/Doug/Hippocampus/HippocampusCodebookFinalPass2.txt'
base_pth = '/home/zach/PythonRepos/PySpots/hybescope_config/'
         
# Import the codebook for genes in the experiment
codewords = pd.read_csv(codebook_pth,  # Warning - file import
                       skiprows=3)
codewords.columns = ['name', 'id', 'barcode']
bcs = []
for bc in codewords.barcode:
    bc = str(bc)
    if len(bc)<nbits:
        bc = '0'*(nbits-len(bc))+bc
    bcs.append(bc)
codewords.barcode = bcs

f = open('/bigstore/GeneralStorage/Rob/merfish/MERFISH_analysis-master/mouse/Doug/Hippocampus/libraryDesign_Hippocampus_final/hippocampus_possible_oligos.fasta', 'r')
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

# cwords = load_codebook(os.path.join(base_pth,'MHD4_18bit_187cwords.csv'))
# # Find unique gene Codewords (isoforms of same gene can have same barcode)
# # and also find unused codewords MHD4 from use codewords for False Positive Detection
# c_dropped = codewords.drop_duplicates('name')
# bc = numpy.array([list(map(int, list(s))) for s in c_dropped.barcode.values])

# blank_codewords = []
# for idx, row in enumerate(distance_matrix(cwords, bc, p=1)):
#     sort_idx = numpy.argsort(row)
#     if row[sort_idx][0]>=4.0:
#         #print(row[sort_idx[:3]])
#         blank_codewords.append(cwords[idx])
# idxes = numpy.random.choice(range(len(blank_codewords)), size=len(cwords)-len(c_dropped.barcode.values), replace=False)
# blank_bc = numpy.array(blank_codewords)[idxes]

# cbook_dict = OrderedDict()
# for idx, row in c_dropped.sort_values('name').iterrows():
#     cbook_dict[row['name']] = numpy.array(list(row.barcode), dtype=float)
    
# blank_dict = {}
# for i, bc in enumerate(blank_bc):
#     blank_dict['blank'+str(i)] = bc
    
# gids, cwords = zip(*cbook_dict.items())
# bids, blanks = zip(*blank_dict.items())
# aids = gids+bids
# gene_codeword_vectors = numpy.stack(cwords, axis=0)
# blank_codeword_vectors = numpy.stack(blanks, axis=0)
# all_codeword_vectors = numpy.concatenate((gene_codeword_vectors,blank_codeword_vectors),axis=0)
# norm_gene_codeword_vectors = normalize(gene_codeword_vectors)
# norm_blank_codeword_vectors = normalize(blank_codeword_vectors)
# norm_all_codeword_vectors = normalize(all_codeword_vectors)

# Import the codebook for genes in the experiment
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
#cwords = np.concatenate([true_cwords,blank_cwords])

gids = list(codebook['name'])
bids = blank_names
aids = gids+bids
gene_codeword_vectors = true_cwords
blank_codeword_vectors = blank_cwords
#all_codeword_vectors = cwords
norm_gene_codeword_vectors = normalize(gene_codeword_vectors)
norm_blank_codeword_vectors = normalize(blank_codeword_vectors)
#norm_all_codeword_vectors = normalize(all_codeword_vectors)

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
parameters['projection_k']=0
parameters['projection_zskip']=1 
parameters['projection_zend']=-1
parameters['projection_function']='None'
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
        
parameters['match_thresh'] = -2
parameters['fpr_thresh'] = 0.2
parameters['nucstain_channel'] = 'DeepBlue'

hotpixel_loc = pickle.load(open('/scratch/hotpixels.pkl','rb'))
# Not certain about hotpixel x vs y
hotpixel_X = hotpixel_loc[0]
hotpixel_Y = hotpixel_loc[1]