import os
from scipy.ndimage import gaussian_filter
from skimage import restoration, io
from scipy import ndimage
import numpy
import numpy as np
from scipy import interpolate
from hybescope_config.microscope_config import *
from fish_results import *
from metadata import Metadata
from functools import partial
import importlib
import multiprocessing
import pickle
import traceback
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str, help="Path to root of imaging folder to initialize metadata.")
    parser.add_argument("cword_config", type=str, help="Path to python file initializing the codewords and providing bitmap variable.")
    parser.add_argument("tforms_path", type=str, help="Path pickle dictionary of tforms per position name.")
    parser.add_argument("out_path", type=str, help="Path to save output.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=8, action='store', help="Number of cores to utilize (default 4).")
    parser.add_argument("-k", type=int, dest="k", default=2, action='store', help="Number z-slices above and below to max project together.")
    parser.add_argument("-s", "--zstart", type=int, dest="zstart", default=4, action='store', help="Start making max projections centered at zstart.")
    parser.add_argument("-m", "--zmax", type=int, dest="zmax", default=15, action='store', help="End making max projections centered at zmax.")
    parser.add_argument("-i", "--zskip", type=int, dest="zskip", default=4, action='store', help="Skip this many z-slices between centers of max projections.")
    parser.add_argument("--decon_iters", type=int, dest="niter", default=25, action='store', help="Skip this many z-slices between centers of max projections.")
    parser.add_argument("--decon_gpu", type=int, dest="gpu", default=0, action='store', help="Skip this many z-slices between centers of max projections.")
    parser.add_argument("--flatfield", type=str, dest="flatfield_path", default='/home/rfor10/repos/pyspots/hybescope_config/flatfields_october2018.pkl', action='store', help="Path to dictionary with flatfield matrix for each channel.")

    args = parser.parse_args()
    
def hdata_multi_z_pseudo_maxprjZ_wrapper(pos_hdata, posname, tforms_xy, tforms_z, md_path, bitmap, cstk_save_dir, reg_ref='hybe1', zstart=5, k=2, zskip=4, zmax=26, ndecon_iter = 20):
    codestacks = {}
    norm_factors = {}
    class_imgs = {}
    for z_i in list(range(zstart, zmax, zskip)):
        try:
            cstk, nf = pseudo_maxproject_positions_and_tform(posname, md_path, tforms_xy, tforms_z, bitmap, zstart=z_i, k=k)
            pos_hdata.add_and_save_data(cstk, posname, z_i, 'cstk')
            pos_hdata.add_and_save_data(nf, posname, z_i, 'nf')
            pos_hdata.add_and_save_data(np.zeros((cstk.shape[0], cstk.shape[1])), posname, z_i, 'cimg')
        except Exception as e:
            pos_hdata.remove_metadata_by_zindex(z_i)
            print(e)
            return 'Failed'
    return 'Passed'
        #codestacks[z_i] = cstk.astype('uint16')
        #norm_factors[z_i] = nf
        #class_imgs[z_i] = np.empty((cstk.shape[0], cstk.shape[1]))
#     if not os.path.exists(cstk_save_dir):
#         os.makedirs(cstk_save_dir)
#     np.savez(os.path.join(cstk_save_dir, posname), cstks=codestacks, 
#             norm_factors = norm_factors, class_imgs = class_imgs)
def pfunc_img_process(img, channel, hybe, t_xy, fltfield, niter=0):
    img = np.divide(zstk, fltfield)
    img = tform_image(img, channel, t_xy, niter=niter)
    return img
def pseudo_maxproject_positions_and_tform(posname, md_path, tforms_xy, tforms_z, bitmap, zstart=6, k=2, reg_ref = 'hybe1', ndecon_iter=22, nf_init_qtile=95):
    """
    Wrapper for multiple Z codestack where each is max_projection of few frames above and below.
    """
    global flatfield_dict, use_gpu
    md = Metadata(md_path)
    xy = tforms_xy
    z = tforms_z
    z = {k: int(np.round(np.mean(v))) for k, v in z.items()}
    z[reg_ref] = 0
    xy[reg_ref] = (0,0)
    seqs, hybes, channels = zip(*bitmap)
    psf_map = {'Orange': orange_psf, 'FarRed': farred_psf, 'Green': green_psf}
    cstk = np.stack([md.stkread(Channel=chan, hybe=hybe, Position=posname, Zindex=list(range(zstart-z[hybe]-k, zstart-z[hybe]+k+1))).max(axis=2) for seq, hybe, chan in bitmap], axis=2)
                      
    if use_gpu:
        inputs = [(cstk[:,:,i], channels[i], hybes[i], xy[hybes[i]], flatfield_dict[channels[i]]) for i in range(cstk.shape[2])]
        cstk = [dogonvole(cstk[:,:,i], psf_map[chan], niter=ndecon_iter) for i, chan in enumerate(channels)]
        with multiprocessing.Pool(24) as ppool:
            cstk = ppool.starmap(pfunc_img_process, inputs)
    else:
        inputs = [(cstk[:,:,i], channels[i], hybes[i], xy[hybes[i]], flatfield_dict[channels[i]], ndecon_iter) for i in range(cstk.shape[2])]
        cstk = ppool.starmap(pfunc_img_process, inputs)
    cstk = np.stack(cstk, axis=2)
    nf = np.percentile(cstk, nf_init_qtile, axis=(0, 1))
    return cstk, nf

def tform_image(cstk, channel, tvect, niter=22):
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
        xs, ys = xshift_db+tvect[1], yshift_db+tvect[0]
        #return cstk.astype('float32')
    if channel == 'Orange':
        xs, ys = xshift_o+tvect[1], yshift_o+tvect[0]
        if niter>0:
            cstk = dogonvole(cstk, orange_psf, niter=niter)
    elif channel=='Green':
        xs, ys = numpy.linspace(0, 2047, 2048)+tvect[1], numpy.linspace(0, 2047, 2048)+tvect[0]
        if niter>0:
            cstk = dogonvole(cstk, green_psf, niter=niter)
    elif channel=='FarRed':
        xs, ys = xshift_fr+tvect[1], yshift_fr+tvect[0]
        if niter>0:
            cstk = dogonvole(cstk, farred_psf, niter=niter)
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

def dogonvole(image, psf, kernel=(2., 2., 0.), blur=(1.1, 1.1, 0.), niter=22):
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
    global hot_pixels, use_gpu, gpu_algorithm
    use_gpu = False
    if not psf.sum() == 1.:
        raise ValueError("psf must be normalized so it sums to 1")
    image = image.astype('float32')
    imin = image.min()
    for y, x in hot_pixels:
        image[y, x] = imin;
        
    img_bg = ndimage.gaussian_filter(image, kernel[:len(image.shape)])
    image = numpy.subtract(image, img_bg)
    numpy.place(image, image<0, 1./2**16)
    image = image.astype('uint16')
    if len(image.shape)==3:
        for i in range(image.shape[2]):
            if use_gpu==1:
                image[:,:,i] = gpu_algorithm.run(fd_data.Acquisition(data=image, kernel=psf), niter=niter).data
            else:
                image[:,:,i] = restoration.richardson_lucy(image[:,:,i], psf,
                                                           niter, clip=False)
    elif len(image.shape)==2:
        if use_gpu==1:
            image = gpu_algorithm.run(fd_data.Acquisition(data=image, kernel=psf), niter=niter).data
        else:
            image = restoration.richardson_lucy(image, psf, niter, clip=False)
    else:
        raise ValueError('image is not a supported dimensionality.')
    image = ndimage.gaussian_filter(image, blur[:len(image.shape)])
    return image

        
if __name__=='__main__':
    niter = args.niter
    md_path = args.md_path
    k = args.k
    zstart = args.zstart
    zskip = args.zskip
    zmax = args.zmax
    out_path = args.out_path
    use_gpu = args.gpu
    ncpu = args.ncpu
    flatfield_path = args.flatfield_path
    flatfield_dict = pickle.load(open(flatfield_path, 'rb'))
    if use_gpu == 1:
        print(device_lib.list_local_devices())
        gpu_algorithm = fd_restoration.RichardsonLucyDeconvolver(2).initialize()
#         assert(ncpu==1, 'If using GPU only use single-threaded to prevent conflicts with GPU usage.')
        os.environ['MKL_NUM_THREADS'] = '16'
        os.environ['GOTO_NUM_THREADS'] = '16'
        os.environ['OMP_NUM_THREADS'] = '16'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['GOTO_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    print(args)
    seqfish_config = importlib.import_module(args.cword_config)
    pfunc = partial(hdata_multi_z_pseudo_maxprjZ_wrapper, md_path=args.md_path, bitmap=seqfish_config.bitmap, k=args.k, zstart=args.zstart, zskip=args.zskip, zmax=args.zmax, cstk_save_dir=args.out_path, ndecon_iter=niter)
    good_positions = pickle.load(open(args.tforms_path, 'rb'))['good']
    func_inputs = []
    for p, t in good_positions.items():
        tforms_xyz = {k: (v[0][0], v[0][1], int(np.round(np.mean(v[0][2])))) for k, v in t.items()}
        txy = {k: (v[0], v[1]) for k, v in tforms_xyz.items()}
        tzz = {k: v[2] for k, v in tforms_xyz.items()}
        func_inputs.append((HybeData(os.path.join(out_path, p)), p, txy, tzz))
    if ncpu>1:
        with multiprocessing.Pool(ncpu) as ppool:
            ppool.starmap(pfunc, func_inputs)
    else:
        for hdata, pos, txy, tzz in func_inputs:
            pfunc(hdata, pos, txy, tzz)