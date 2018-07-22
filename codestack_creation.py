from skimage import io
import os
from skimage.filters import gaussian_filter
from skimage import restoration
import numpy
import numpy as np
from scipy import interpolate


def interp_warp(img, x, y):
    """
    Apply chromatic abberation shifts to images.
    
    Parameters
    ----------
    img : ndarray
    x : array
    y : array
    
    Returns
    -------
    nimg : ndarray - same size as img but interpolated from x, y onto 0, 1, ... , img.shape
    """
    i2 = interpolate.interp2d(x, y, img)
    nimg = i2(range(img.shape[0]), range(img.shape[1]))
    return nimg

def dogonvole(image, psf, kernel=(2.1, 2.1, 0.), blur=(1.1, 1.1, 0.), niter=8):
    """
    Perform deconvolution and difference of gaussian processing.
    
    Parameters
    ----------
    image : ndarray
    psf : ndarray
    kernel : tuple
    blur : tuple
    niter : int
    
    Returns
    -------
    image : ndarray
        Processed image same shape as image input.
    """
    if not psf.sum() == 1.:
        raise ValueError("psf must be normalized so it sums to 1")
    image = image.astype('float32')
    img_bg = gaussian_filter(image, kernel[:len(image.shape)])
    image = numpy.subtract(image, img_bg)
    numpy.place(image, image<0, 1./2**16)
    image = image.astype('uint16')
    if len(image.shape)==3:
        for i in range(image.shape[2]):
            image[:,:,i] = restoration.richardson_lucy(image[:,:,i], psf, niter, clip=False)
    elif len(image.shape)==2:
        image = restoration.richardson_lucy(image, psf, niter, clip=False)
    else:
        raise ValueError('image is not a supported dimensionality.')
    image = gaussian_filter(image, blur[:len(image.shape)])
    return image

def process_image_h5py(fname, min_thresh=5./2**16, niter=15, uint8=False):
    """
    WARNING - this function is written in a computer specific way to 
    save to the scratch drive of the hybe_computer. Probably needs to be 
    rethought. Images move from bigstore to scratch during processing because 
    scratch is SSD and can be IO bound.
    
    Perform on-the-fly initial processing of images.
    
    Parameters
    ----------
    fname : str
        Filename of a tiff image to process
    min_thresh : float
        Minimum intensity will become 0 in output
    niter : int
        Number of iterations of deconvolution to perform.
    
    Returns
    -------
    Nothing but new file is written
    
    Notes - currently save locations are hardcoded inside here.
    """
    # This .pkl file needs to be in the spot_calling repository/directory
    fnames_parts = fname.split('/')
    fname_out = os.path.join('/scratch', *fnames_parts[2:])
    #print(fname_out)
    if os.path.isfile(fname_out):
        #print('Already exists', fname_out)
        return
    cstk = io.imread(fname).astype('float32')#/2.**16
    
    if 'DeepBlue' in fname:
#         xs, ys = yshift_db+tvect[0], xshift_db+tvect[1]
        return cstk.astype('uint8')
    if 'Orange' in fname:
        #xs, ys = numpy.linspace(0, 2047, 2048)+tvect[0], numpy.linspace(0, 2047, 2048)+tvect[1]
        cstk = dogonvole(cstk, orange_psf, niter=niter)
    elif 'Green' in fname:
        #xs, ys = yshift_g+tvect[0], xshift_g+tvect[1]
        cstk = dogonvole(cstk, green_psf, niter=niter)
    elif 'FarRed' in fname:
        #xs, ys = yshift_fr+tvect[0], xshift_fr+tvect[1]
        cstk = dogonvole(cstk, farred_psf, niter=niter)
    else:
        print('unmet name')
        return
    #cstk = interp_warp(cstk, xs, ys)
    #pdb.set_trace()
    
    #stkshow(cstk.astype('uint8'), fname='/home/rfor10/Downloads/stkio.tif')
    #return cstk.astype('uint8')
    fnames_parts = fname.split('/')
    fname_out = os.path.join('/scratch', *fnames_parts[2:])
    if not os.path.exists(os.path.split(fname_out)[0]):
        os.makedirs(os.path.split(fname_out)[0])
    #print(fname_out, fnames_parts)
    if uint8:
        cstk = cstk/2**6
        io.imsave(fname_out, cstk.astype('uint8'))
    else:
        io.imsave(fname_out, cstk.astype('uint16'))

        
def process_image_postimaging(fname, min_thresh=5./2**16, niter=15, uint8=False):
    """    
    Parameters
    ----------
    fname : str
        Filename of a tiff image to process
    min_thresh : float
        Minimum intensity will become 0 in output
    niter : int
        Number of iterations of deconvolution to perform.
    
    Returns
    -------
    Nothing but new file is written
    
    Notes - currently save locations are hardcoded inside here.
    """
    cstk = io.imread(fname).astype('float32')#/2.**16
    
    if 'DeepBlue' in fname:
#         xs, ys = yshift_db+tvect[0], xshift_db+tvect[1]
        return cstk.astype('uint8')
    if 'Orange' in fname:
        #xs, ys = numpy.linspace(0, 2047, 2048)+tvect[0], numpy.linspace(0, 2047, 2048)+tvect[1]
        cstk = dogonvole(cstk, orange_psf, niter=niter)
    elif 'Green' in fname:
        #xs, ys = yshift_g+tvect[0], xshift_g+tvect[1]
        cstk = dogonvole(cstk, green_psf, niter=niter)
    elif 'FarRed' in fname:
        #xs, ys = yshift_fr+tvect[0], xshift_fr+tvect[1]
        cstk = dogonvole(cstk, farred_psf, niter=niter)
    else:
        print('unmet name')
        return
    if uint8:
        cstk = cstk/2**6
        io.imsave(fname, cstk.astype('uint8'))
    else:
        io.imsave(fname, cstk.astype('uint16'))
        
# Warning function below uses global variables
def tform_image(cstk, channel, tvect):
    """
    Warp images to correct chromatic abberation and translational stage drift.
    
    Parameters
    ----------
    cstk : ndarray
        Image(s) to be warped
    channel : str
        Name of channel (used to determine which chromatic warping to apply)
    tvect : tuple
        Tuple of floats to correct for stage drift
        
    Returns
    -------
    cstk : ndarray float32
        Warped image of same shape as input cstk
        
    Notes - Chromatic abberation maps are imported from seqfish_config and as accessed as globals
    """
    #cstk = pdata.getImage(str(ix))
    #cstk = io.imread(fname).astype('float64')#/2.**16
    if channel == 'DeepBlue':
        xs, ys = yshift_db+tvect[1], xshift_db+tvect[0]
        #return cstk.astype('float32')
    if channel == 'Orange':
        xs, ys = numpy.linspace(0, 2047, 2048)+tvect[1], numpy.linspace(0, 2047, 2048)+tvect[0]
        #cstk = dogonvole(cstk, orange_psf, niter=niter)
    elif channel=='Green':
        xs, ys = yshift_g+tvect[1], xshift_g+tvect[0]
        #cstk = dogonvole(cstk, green_psf, niter=niter)
    elif channel=='FarRed':
        xs, ys = yshift_fr+tvect[1], xshift_fr+tvect[0]
        #cstk = dogonvole(cstk, farred_psf, niter=niter)
    cstk = interp_warp(cstk, xs, ys)
    return cstk.astype('float32')

from seqfish_config import *
def pseudo_maxproject_positions_and_tform(posname, tforms_xy, tforms_z, zstart=6,
                                          k=2, reg_ref = 'hybe1'):
    """
    Wrapper for multiple Z codestack where each is max_projection of few frames above and below.
    """
    md = Metadata(md_pth)
    xy = tforms_xy[posname]
    z = tforms_z[posname]
    z = {k: int(np.round(np.mean(v))) for k, v in z.items()}
    z[reg_ref] = 0
    xy[reg_ref] = (0,0)
    cstk = []
    for seq, hybe, chan in bitmap:
        t = xy[hybe]
        zindexes = list(range(zstart-z[hybe]-k, zstart-z[hybe]+k+1))
        print(zindexes)
        zstk = md.stkread(Channel=chan, hybe=hybe, Position=posname, Zindex=zindexes)
        zstk = zstk.max(axis=2)
        zstk = tform_image(zstk, chan, t)
        cstk.append(zstk)
        del zstk
    cstk = np.stack(cstk, axis=2)
    nf = np.percentile(cstk, 90, axis=(0, 1))
    return cstk, nf
