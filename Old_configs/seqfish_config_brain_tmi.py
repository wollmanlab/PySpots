import numpy
import numpy as np
import pickle
import pandas
import os
from scipy.spatial import distance_matrix
from collections import OrderedDict
from sklearn.preprocessing import normalize

# Basic parameters of imaging

bitmap = [('RS0095_cy5', 'hybe2', 'FarRed'), ('RS0109_cy5', 'hybe4', 'FarRed'),
          ('RS0175_cy5', 'hybe6', 'FarRed'), ('RS0237_cy5', 'hybe1', 'FarRed'),
          ('RS0307_cy5', 'hybe3', 'FarRed'), ('RS0332_cy5', 'hybe5', 'FarRed'),
          ('RS0384_atto565', 'hybe5', 'Orange'), ('RS0406_atto565', 'hybe6', 'Orange'),
          ('RS0451_atto565', 'hybe4', 'Orange'), ('RS0468_atto565', 'hybe3', 'Orange'),
          ('RS0548_atto565', 'hybe2', 'Orange'), ('RS64.0_atto565', 'hybe1', 'Orange'),
          ('RSN9927.0_cy5', 'hybe8', 'FarRed'), ('RSN2336.0_cy5', 'hybe7', 'FarRed'), 
          ('RSN1807.0_cy5', 'hybe9', 'FarRed'), ('RSN4287.0_atto565', 'hybe7', 'Orange'), 
          ('RSN1252.0_atto565', 'hybe9', 'Orange'), ('RSN9535.0_atto565', 'hybe8', 'Orange')
         ]
nbits = len(bitmap)

# codebook_pth = '/bigstore/GeneralStorage/Rob/merfish/MERFISH_analysis-master/mouse/mouse_brain_trauma/TBIfinalCodebook.txt'
# codewords = pandas.read_csv(codebook_pth,  # Warning - file import
#                        skiprows=3)
# codewords.columns = ['name', 'id', 'barcode']
# bcs = []
# for bc in codewords.barcode:
#     bc = str(bc)
#     if len(bc)<nbits:
#         bc = '0'*(nbits-len(bc))+bc
#     bcs.append(bc)
# codewords.barcode = bcs


# f = open('/home/rfor10/repos/seqfish_design/ca_celltype_v2/combined_oligos.fa', 'r')
# s = f.read()
# f.close()
# present = [i in s for i in codewords.id.values]
# codewords = codewords[present]

def load_codebook(fname):
    barcodes = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            bc = map(int, line.strip().split(','))
            barcodes.append(list(bc))
    return np.array(barcodes)

codewords = pickle.load(open('/home/rfor10/repos/seqfish_design/mouse_tbi_codewords.pkl', 'rb'))
gids, codewords = zip(*codewords)
bc = np.array(codewords)

cwords = load_codebook('/home/rfor10/repos/seqfish_design/MHD4_18bit_187cwords.csv')
# Find unique gene Codewords (isoforms of same gene can have same barcode)
# and also find unused codewords MHD4 from use codewords for False Positive Detection
# c_dropped = codewords.drop_duplicates('name')
# bc = numpy.array([list(map(int, list(s))) for s in c_dropped.barcode.values])

blank_codewords = []
for idx, row in enumerate(distance_matrix(cwords, bc, p=1)):
    sort_idx = numpy.argsort(row)
    if row[sort_idx][0]>=4.0:
        #print(row[sort_idx[:3]])
        blank_codewords.append(cwords[idx])
idxes = numpy.random.choice(range(len(blank_codewords)), size=len(blank_codewords), replace=False)
blank_bc = numpy.array(blank_codewords)[idxes]

# cbook_dict = OrderedDict()
# for idx, row in c_dropped.sort_values('name').iterrows():
#     cbook_dict[row['name']] = numpy.array(list(row.barcode), dtype=float)
    
blank_dict = {}
for i, bc in enumerate(blank_bc):
    blank_dict['blank'+str(i)] = bc
    
bids, blanks = zip(*blank_dict.items())
gene_codeword_vectors = numpy.stack(codewords, axis=0)
blank_codeword_vectors = numpy.stack(blanks, axis=0)
norm_gene_codeword_vectors = normalize(gene_codeword_vectors)
norm_blank_codeword_vectors = normalize(blank_codeword_vectors)
