import pickle
import numpy
import os
from skimage.io import imread

################################ Microscope Related Config ##########################
image_size = (2048, 2048)

base_pth = os.path.dirname(__file__) # finds location of this module
ave_bead = pickle.load(open(os.path.join(base_pth, 'ave_bead.333um.pkl'), 'rb'))
hot_pixels = pickle.load(open(os.path.join(base_pth, 'hot_pixels_aug2018.pkl'), 'rb'))

# Pre october 2018
# chromatic_dict = pickle.load(open(os.path.join(base_pth, 'jan2018_chromatic.pkl'), 'rb')) # Warning File import

# xshift_fr = numpy.add(chromatic_dict['orange_minus_farred'][0], range(image_size[0]))
# yshift_fr = numpy.add(chromatic_dict['orange_minus_farred'][1], range(image_size[1]))

# xshift_g = numpy.add(chromatic_dict['orange_minus_green'][0], range(image_size[0]))
# yshift_g = numpy.add(chromatic_dict['orange_minus_green'][1], range(image_size[1]))

# xshift_db = numpy.add(chromatic_dict['orange_minus_deepblue'][0], range(image_size[0]))
# yshift_db = numpy.add(chromatic_dict['orange_minus_deepblue'][1], range(image_size[1]))
# Post october 2018
# Green - Other Channel
chromatic_dict = pickle.load(open(os.path.join(base_pth, 'chromatic_october2018.pkl'), 'rb'))

xshift_fr = numpy.add(chromatic_dict['FarRed']['x'].mean(axis=0), range(image_size[0]))
yshift_fr = numpy.add(chromatic_dict['FarRed']['y'].mean(axis=1), range(image_size[1]))

xshift_o = numpy.add(chromatic_dict['Orange']['x'].mean(axis=0), range(image_size[0]))
yshift_o = numpy.add(chromatic_dict['Orange']['y'].mean(axis=1), range(image_size[1]))

xshift_db = numpy.add(chromatic_dict['DeepBlue']['x'].mean(axis=0), range(image_size[0]))
yshift_db = numpy.add(chromatic_dict['DeepBlue']['y'].mean(axis=1), range(image_size[1]))

farred_psf = imread(os.path.join(base_pth, 'farred_psf_fit_250nmZ_63x.tif'))
farred_psf = farred_psf[25, 8:17, 8:17]
farred_psf = farred_psf/farred_psf.sum()
green_psf = imread(os.path.join(base_pth, 'green_psf_fit_250nmZ_63x.tif'))
green_psf = green_psf[28, 5:14, 5:14]
green_psf = green_psf/green_psf.sum()
orange_psf = imread(os.path.join(base_pth, 'orange_psf_fit_250nmZ_63x.tif'))
orange_psf = orange_psf[25,  5:14, 5:14]
orange_psf = orange_psf/orange_psf.sum()
