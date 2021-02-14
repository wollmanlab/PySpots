import pickle
import numpy
import os
from skimage.io import imread

################################ Microscope Related Config ##########################
image_size = (2048, 2048)

base_pth = os.path.dirname(__file__) # finds location of this module
ave_bead = pickle.load(open(os.path.join(base_pth, 'ave_bead.333um.pkl'), 'rb'))
hot_pixels = pickle.load(open(os.path.join(base_pth, 'hot_pixels_oct2018.pkl'), 'rb'))
flatfield_dict = pickle.load(open(os.path.join(base_pth, 'flatfields_october2018.pkl'), 'rb'))

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

xshift_fr = chromatic_dict['FarRed']['x']
yshift_fr = chromatic_dict['FarRed']['y']

xshift_o = chromatic_dict['Orange']['x']
yshift_o = chromatic_dict['Orange']['y']

xshift_db = chromatic_dict['DeepBlue']['x']
yshift_db = chromatic_dict['DeepBlue']['y']

chromatic_dict['Green'] = {}
chromatic_dict['Green']['x'] = chromatic_dict['DeepBlue']['x']
chromatic_dict['Green']['y'] = chromatic_dict['DeepBlue']['y']



farred_psf = imread(os.path.join(base_pth, 'farred_psf_fit_250nmZ_63x.tif'))
farred_psf = farred_psf[25, 8:17, 8:17]
farred_psf = farred_psf/farred_psf.sum()
green_psf = imread(os.path.join(base_pth, 'green_psf_fit_250nmZ_63x.tif'))
green_psf = green_psf[28, 5:14, 5:14]
green_psf = green_psf/green_psf.sum()
orange_psf = imread(os.path.join(base_pth, 'orange_psf_fit_250nmZ_63x.tif'))
orange_psf = orange_psf[25,  5:14, 5:14]
orange_psf = orange_psf/orange_psf.sum()

farred_psf_3d = imread(os.path.join(base_pth, 'farred_psf_fit_250nmZ_63x.tif'))
farred_psf_3d = farred_psf_3d[20:29, 8:17, 8:17]
farred_psf_3d = farred_psf_3d/farred_psf_3d.sum()
green_psf_3d = imread(os.path.join(base_pth, 'green_psf_fit_250nmZ_63x.tif'))
green_psf_3d = green_psf_3d[20:29, 5:14, 5:14]
green_psf_3d = green_psf_3d/green_psf_3d.sum()
orange_psf_3d = imread(os.path.join(base_pth, 'orange_psf_fit_250nmZ_63x.tif'))
orange_psf_3d = orange_psf_3d[20:29,  5:14, 5:14]
orange_psf_3d = orange_psf_3d/orange_psf_3d.sum()

psf_dict = {'FarRed':farred_psf,'Orange':orange_psf,'Green':green_psf}
psf_dict_3d = {'FarRed':farred_psf_3d,'Orange':orange_psf_3d,'Green':green_psf_3d}