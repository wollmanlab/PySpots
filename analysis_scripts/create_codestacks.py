from skimage import io
import os
from scipy.ndimage import gaussian_filter
from skimage import restoration
import numpy
import numpy as np
from scipy import interpolate
from hybescope_config.microscope_config import *
from metadata import Metadata
from functools import partial


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("md_path", type=str, help="Path to root of imaging folder to initialize metadata.")
parser.add_argument("tforms_pth", type=str, help="Path pickle dictionary of tforms per position name.")
parser.add_argument("out_path", type=str, help="Path to save output.")
parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=4, action='store', nargs=1, help="Number of cores to utilize (default 4).")

args = parser.parse_args()


def multi_z_pseudo_maxprjZ_wrapper(posname, tforms_xy, tforms_z,
                                   reg_ref='hybe1', zstart=5,
                                   k=2, zskip=4, zmax=26):
    global cstk_save_dir
    codestacks = {}
    norm_factors = {}
    class_imgs = {}
    for z_i in list(range(zstart, zmax, zskip)):
        cstk, nf = pseudo_maxproject_positions_and_tform(posname, tforms_xy,
                                                         tforms_z, zstart=z_i, 
                                                        k=k)
        codestacks[z_i] = cstk.astype('uint16')
        norm_factors[z_i] = nf
        class_imgs[z_i] = np.empty((cstk.shape[0], cstk.shape[1]))
    np.savez(os.path.join(cstk_save_dir, posname), cstks=codestacks, 
            norm_factors = norm_factors, class_imgs = class_imgs)

def pseudo_maxproject_positions_and_tform(posname, tforms_xy, tforms_z, zstart=6,
                                          k=2, reg_ref = 'hybe1'):
    """
    Wrapper for multiple Z codestack where each is max_projection of few frames above and below.
    """
    md = Metadata(md_pth)
    xy = tforms_xy
    z = tforms_z
    z = {k: int(np.round(np.mean(v))) for k, v in z.items()}
    z[reg_ref] = 0
    xy[reg_ref] = (0,0)
    cstk = []
    for seq, hybe, chan in bitmap:
        t = xy[hybe]
        zindexes = list(range(zstart-z[hybe]-k, zstart-z[hybe]+k+1))
        print(zindexes)
        zstk = md.stkread(Channel=chan, hybe=hybe,
                          Position=posname, Zindex=zindexes)
        zstk = zstk.max(axis=2)
        zstk = tform_image(zstk, chan, t)
        cstk.append(zstk)
        del zstk
    cstk = np.stack(cstk, axis=2)
    nf = np.percentile(cstk, 90, axis=(0, 1))
    return cstk, nf
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

if __name__=='__main__':
    cstk_save_dir = out_path
    if cstk_save_dir[-1] == '/':
        cstk_save_dir = cstk_save_dir[:-1]
    if md_path[-1] == '/':
        md_path = md_path[:-1]
    pfunc = partial(multi_z_pseudo_maxprjZ_wrapper, k=1, zstart=3, zskip=3, zmax=13)
    good_positions = pickle.load(open(tforms_pth, 'rb'))
    func_inputs = []
    for p, (t, q) in good_positions.items():
        tforms_xyz = {k: (v[0], v[1], int(np.round(np.mean(v[2])))) for k, v in t.items()}
        txy = {k: (v[0], v[1]) for k, v in tforms_xyz.items()}
        tzz = {k: v[2] for k, v in tforms_xyz.items()}
        func_inputs.append((p, txy, tzz))
    with multiprocessing.Pool(ncpu) as ppool:
        ppool.starmap(pfunc, func_inputs)