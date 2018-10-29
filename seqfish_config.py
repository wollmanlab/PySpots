
# Config for analysis of SeqFish data in Wollman lab (Jan 2018)
# Necessary files are assumed to be in the current directory when this config is imported.
# Currently designed for Python 3.5

################################ Probe/Codebook Related Config ##########################
# This bitmap is tyically constant for most experiments
# Might need to change this if experimental conditions are different
import numpy
import pickle
import pandas
import os
from scipy.spatial import distance_matrix
from skimage.io import imread
from collections import OrderedDict
from sklearn.preprocessing import normalize

# Basic parameters of imaging
image_size = (2048, 2048)

bitmap = [('RS0095', 'hybe2', 'FarRed'), ('RS0109', 'hybe4', 'FarRed'),
          ('RS0175', 'hybe6', 'FarRed'), ('RS0237', 'hybe1', 'FarRed'),
          ('RS0307', 'hybe3', 'FarRed'), ('RS0332', 'hybe5', 'FarRed'),
          ('RS0384', 'hybe5', 'Orange'), ('RS0406', 'hybe6', 'Orange'),
          ('RS0451', 'hybe4', 'Orange'), ('RS0468', 'hybe3', 'Orange'),
          ('RS0548', 'hybe2', 'Orange'), ('RS64.0', 'hybe1', 'Orange'),
          ('RS156.0', 'hybe3', 'Green'), ('RS278.0', 'hybe4','Green'),
          ('RS313.0', 'hybe5', 'Green'), ('RS643.0', 'hybe1', 'Green'),
          ('RS740.0', 'hybe2', 'Green'), ('RS810.0', 'hybe6', 'Green')
         ]
nbits = len(bitmap)

# config_options
codebook_pth = '/home/rfor10/repos/spot_calling/calcium_codebook_final.csv'
base_pth = '/home/rfor10/repos/spot_calling'
         
# Import the codebook for genes in the experiment
codewords = pandas.read_csv(codebook_pth,  # Warning - file import
                       skiprows=3)
codewords.columns = ['name', 'id', 'barcode']
bcs = []
for bc in codewords.barcode:
    bc = str(bc)
    if len(bc)<nbits:
        bc = '0'*(nbits-len(bc))+bc
    bcs.append(bc)
codewords.barcode = bcs

# Find unique gene Codewords (isoforms of same gene can have same barcode)
# and also find unused codewords MHD4 from use codewords for False Positive Detection
c_dropped = codewords.drop_duplicates('name')
bc = numpy.array([list(map(int, list(s))) for s in c_dropped.barcode.values])

cwords = pickle.load(open(os.path.join(base_pth, 'cbook.pkl'), 'rb'), encoding='latin1') # Warning - file import

blank_codewords = []
for idx, row in enumerate(distance_matrix(cwords, bc, p=1)):
    sort_idx = numpy.argsort(row)
    if row[sort_idx][0]>=4.0:
        #print(row[sort_idx[:3]])
        blank_codewords.append(cwords[idx])
idxes = numpy.random.choice(range(len(blank_codewords)), size=12, replace=False)
blank_bc = numpy.array(blank_codewords)[idxes]

cbook_dict = OrderedDict()
for idx, row in c_dropped.sort_values('name').iterrows():
    cbook_dict[row['name']] = numpy.array(list(row.barcode), dtype=float)
    
blank_dict = {}
for i, bc in enumerate(blank_bc):
    blank_dict['blank'+str(i)] = bc

gids, cwords = zip(*cbook_dict.items())
bids, blanks = zip(*blank_dict.items())
gene_codeword_vectors = numpy.stack(cwords, axis=0)
blank_codeword_vectors = numpy.stack(blanks, axis=0)
norm_gene_codeword_vectors = normalize(gene_codeword_vectors)
norm_blank_codeword_vectors = normalize(blank_codeword_vectors)

################################ Microscope Related Config ##########################
ave_bead = pickle.load(open(os.path.join(base_pth, 'ave_bead.333um.pkl'), 'rb'))

# This .pkl file needs to be in the spot_calling repository/directory
chromatic_dict = pickle.load(open(os.path.join(base_pth, './jan2018_chromatic.pkl'), 'rb')) # Warning File import

xshift_fr = numpy.add(chromatic_dict['orange_minus_farred'][0], range(image_size[0]))
yshift_fr = numpy.add(chromatic_dict['orange_minus_farred'][1], range(image_size[1]))

xshift_g = numpy.add(chromatic_dict['orange_minus_green'][0], range(image_size[0]))
yshift_g = numpy.add(chromatic_dict['orange_minus_green'][1], range(image_size[1]))

xshift_db = numpy.add(chromatic_dict['orange_minus_deepblue'][0], range(image_size[0]))
yshift_db = numpy.add(chromatic_dict['orange_minus_deepblue'][1], range(image_size[1]))

# dx = chromatic_dict['dx_model_oMfarred']
# dy = chromatic_dict['dy_model_oMfarred']
# xshift = numpy.add(range(2048), numpy.polyval(dx, range(2048)))
# yshift = numpy.add(range(2048), numpy.polyval(dy, range(2048)))
# xshift_fr, yshift_fr = numpy.meshgrid(xshift, yshift, sparse=True)

# dx = chromatic_dict['dx_model_oMgreen']
# dy = chromatic_dict['dy_model_oMgreen']
# xshift = numpy.add(range(2048), numpy.polyval(dx, range(2048)))
# yshift = numpy.add(range(2048), numpy.polyval(dy, range(2048)))
# xshift_g, yshift_g = numpy.meshgrid(xshift, yshift, sparse=True)

farred_psf = imread(os.path.join(base_pth, 'farred_psf_fit_250nmZ_63x.tif'))
farred_psf = farred_psf[25, 8:17, 8:17]
farred_psf = farred_psf/farred_psf.sum()
green_psf = imread(os.path.join(base_pth, 'green_psf_fit_250nmZ_63x.tif'))
green_psf = green_psf[28, 5:14, 5:14]
green_psf = green_psf/green_psf.sum()
orange_psf = imread(os.path.join(base_pth, '63x_psf_orange_250nmZ.tif'))
orange_psf = orange_psf[25,  5:14, 5:14]
orange_psf = orange_psf/orange_psf.sum()