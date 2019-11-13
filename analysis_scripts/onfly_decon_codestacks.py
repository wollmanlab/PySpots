#!/usr/bin/env python
import os
import sys
from skimage.filters import gaussian
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
import dill as pickle
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str, help="Path to root of imaging folder to initialize metadata.")
    parser.add_argument("cword_config", type=str, help="Name of python module defining bitmap information.")
    parser.add_argument("tforms_path", type=str, help="Path pickle dictionary of tforms per position name.")
    parser.add_argument("out_path", type=str, help="Path to save output.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=8, action='store', help="Number of cores to utilize (default 4).")
    parser.add_argument("-k", type=int, dest="k", default=2, action='store', help="Number z-slices above and below to max project together.")
    parser.add_argument("-s", "--zstart", type=int, dest="zstart", default=4, action='store', help="Start making max projections centered at zstart.")
    parser.add_argument("-m", "--zmax", type=int, dest="zmax", default=15, action='store', help="End making max projections centered at zmax.")
    parser.add_argument("-i", "--zskip", type=int, dest="zskip", default=4, action='store', help="Skip this many z-slices between centers of max projections.")
    parser.add_argument("--decon_iters", type=int, dest="niter", default=10, action='store', help="Skip this many z-slices between centers of max projections.")
    parser.add_argument("--decon_gpu", type=int, dest="gpu", default=0, action='store', help="Skip this many z-slices between centers of max projections.")
    parser.add_argument("--bg_sigma", type=float, dest="kernel_sigma", default=2., action='store', help="Background subtraction sigma")
    parser.add_argument("--blur_sigma", type=float, dest="blur_sigma", default=0.9, action='store', help="Blur sigma")
    parser.add_argument("--flatfield", type=str, dest="flatfield_path", default='/home/rfor10/repos/pyspots/hybescope_config/flatfields_october2018.pkl', action='store', help="Path to dictionary with flatfield matrix for each channel.")

    args = parser.parse_args()
    
def hdata_multi_z_pseudo_maxprjZ_wrapper(pos_hdata, posname, tforms_xy, tforms_z, md, bitmap, reg_ref='hybe1',
                                         zstart=5, k=2, zskip=4, zmax=26, ndecon_iter = 10, nf_init_qtile=95,
                                         prj_func='mean', use_gpu=1):
    global image_size, out_path, flatfield_dict, global_kernel,global_blur, psf_map
    NoneType = type(None)
    tforms_z = {k: int(np.round(np.mean(v))) for k, v in tforms_z.items()}
    tforms_z[reg_ref] = 0
    tforms_xy[reg_ref] = (0,0)
    seqs, hybes, channels = zip(*bitmap)
    highest_z = md.image_table.Zindex.max()
#     psf_map = {'Orange': orange_psf, 'FarRed': farred_psf, 'Green': green_psf}
    hybes, channels = np.array(hybes), np.array(channels)
    completed_idx = np.array([i for i in range(len(hybes)) if hybes[i] in tforms_xy])
    if not os.path.exists(os.path.join(pos_hdata.base_path, 'processing.pkl')):
        completed_not_processed_idx = completed_idx
    else:
        print('Some done')
        doneso = pickle.load(open(os.path.join(pos_hdata.base_path, 'processing.pkl'), 'rb'))
        completed_not_processed_idx = np.array([i for i in completed_idx if i not in doneso])
    if len(completed_not_processed_idx) == 0:
        return 'Passed'

    print('\n','Imaging completed: ', completed_idx)
    pool = multiprocessing.Pool(18)
    for z_i in list(range(zstart, zmax, zskip)):
        print(z_i)
        try:
            cstk = pos_hdata.load_data(posname, z_i, 'cstk')
            if isinstance(cstk,NoneType):
                cstk = np.empty((image_size[0], image_size[1], len(bitmap)))
                completed_not_processed_idx = completed_idx
            else:
                completed_not_processed_idx = np.array([i for i in completed_idx if np.mean(cstk[:,:,i])==0])
            print('Imaging done but not processed: ', completed_not_processed_idx)
            if len(completed_not_processed_idx) == 0:
                continue
            codebits = []
            t_xys = []
            for bitmap_idx in completed_not_processed_idx:
                seq, hybe, chan = bitmap[bitmap_idx]
                t_z = tforms_z[hybe]
                t_xy = tforms_xy[hybe]
                t_xys.append(t_xy)
                if any(np.array(list(range(z_i-t_z-k, z_i-t_z+k+1)))<=highest_z) or any(np.array(list(range(z_i-t_z-k, z_i-t_z+k+1)))>=0):
                    local_z = np.array(list(range(z_i-t_z-k, z_i-t_z+k+1)))
                    local_z = local_z[np.where(local_z<=highest_z)[0]].tolist()
                    cbit = md.stkread(Channel=chan, hybe=hybe, Position=posname, Zindex=local_z) # z info not preprocessed?
                    cbit = dogonvole(cbit, psf_map[chan], gpu_algorithm, niter=ndecon_iter,kernel=global_kernel, blur=global_blur, use_gpu=use_gpu)
                else:
                    print('No images at that Zrange')
                    cbit = np.ones((2048, 2048, 3)) # hardcoding bad!
                if prj_func == 'max':
                    cbit = cbit.max(axis=2)
                elif prj_func == 'mean':
                    cbit = cbit.mean(axis=2)
#                 else:
#                     raise ValueError("prj_func type not recognized as implemented."
#                 cbit = dogonvole(cbit, psf_map[chan], gpu_algo, niter=ndecon_iter, blur=(smooth_kernel, smooth_kernel, 0)) # MERGE Conflict 9/9/19
#                 cbit = np.divide(cbit, flatfield_dict[channels[bitmap_idx]])
#                 cbit = tform_image(cbit, chan, t_xy, niter=niter)
                
                codebits.append(cbit)
            hnames= hybes[completed_not_processed_idx]
            channels_subset = channels[completed_not_processed_idx]
            nf = np.empty(len(bitmap))
            sys.stdout.flush()
            results = pool.starmap(pfunc_img_process, zip(codebits, channels_subset, hnames, t_xys))
            for bitmap_idx, cbit in zip(completed_not_processed_idx, results):
                cstk[:,:,bitmap_idx] = cbit
            nf = np.percentile(cstk, nf_init_qtile, axis=(0, 1))
            pos_hdata.add_and_save_data(cstk, posname, z_i, 'cstk')
            pos_hdata.add_and_save_data(nf, posname, z_i, 'nf')
            pos_hdata.add_and_save_data(np.zeros((cstk.shape[0], cstk.shape[1])), posname, z_i, 'cimg')
            
        except Exception as e:
            print(posname,z_i,'Failed')
            print(e)
            print(chan, hybe, posname, list(range(z_i-t_z-k, z_i-t_z+k+1)))
            pdb.set_trace()
            return 'Failed'
    try:
        if not os.path.exists(pos_hdata.base_path):
            print('Warning: had to create directory structure so the loop of Z probably did not work.')
            os.makedirs(pos_hdata.base_path)
        pickle.dump(set(completed_idx), open(os.path.join(pos_hdata.base_path, 'processing.pkl'), 'wb'))
    except Exception as e:
        print(posname,'Entire Position Failed')
        print(e)
    pool.close()
    sys.stdout.flush()
    return 'Passed'

def pfunc_img_process(img, channel, hybe, t_xy, niter=0):
    global flatfield_dict                                                 
    img = np.divide(img, flatfield_dict[channel])
    img = tform_image(img, channel, t_xy, niter=niter)
    return img

def tform_image(img, channel, tvect, niter=20):
    """
    Warp images to correct chromatic abberation and translational stage drift.
    
    Parameters
    ----------
    img : ndarray
        Image(s) to be warped
    channel : str
        Name of channel (used to determine which chromatic warping to apply)
    tvect : tuple
        Tuple of floats to correct for stage drift
        
    Returns
    -------
    img : ndarray float32
        Warped image of same shape as input img
        
    Notes - Chromatic abberation maps are imported from seqfish_config and as accessed as globals
    """
    if channel == 'DeepBlue':
        xs, ys = xshift_db+tvect[1], yshift_db+tvect[0]
        #return cstk.astype('float32')
    elif channel == 'Orange':
        xs, ys = xshift_o+tvect[1], yshift_o+tvect[0]
        if niter>0:
            img = dogonvole(img, orange_psf, niter=niter)
    elif channel=='Green':
        xs, ys = numpy.linspace(0, 2047, 2048)+tvect[1], numpy.linspace(0, 2047, 2048)+tvect[0]
        if niter>0:
            img = dogonvole(img, green_psf, niter=niter)
    elif channel=='FarRed':
        xs, ys = xshift_fr+tvect[1], yshift_fr+tvect[0]
        if niter>0:
            img = dogonvole(img, farred_psf, niter=niter)
    img = interp_warp(img, xs, ys)
    return img.astype('float32')

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

def dogonvole(image, psf, gpu_algorithm, kernel=(2., 2., 0.), blur=(0.9, 0.9, 0.), niter=10, use_gpu=1):
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
    global hot_pixels

    if not psf.sum() == 1.:
        raise ValueError("psf must be normalized so it sums to 1")
    image = image.astype('float32')
    imin = image.min()
    for y, x in hot_pixels:
        image[y, x] = imin;
        
    img_bg = gaussian(image, kernel[:len(image.shape)], preserve_range=True, multichannel=False)
    image = numpy.subtract(image, img_bg)
    numpy.place(image, image<0, 1./2**16)
    image = image.astype('uint16')
    if len(image.shape)==3:
        for i in range(image.shape[2]):
            if use_gpu==1:
                image[:,:,i] = gpu_algorithm.run(fd_data.Acquisition(data=image[:,:,i], kernel=psf), niter=niter).data
            else:
                image[:,:,i] = restoration.richardson_lucy(image[:,:,i], psf,niter, clip=False)
    elif len(image.shape)==2:
        if use_gpu==1:
            image = gpu_algorithm.run(fd_data.Acquisition(data=image, kernel=psf), niter=niter).data
        else:
            image = restoration.richardson_lucy(image, psf, niter, clip=False)
    else:
        raise ValueError('image is not a supported dimensionality.')
    image = gaussian(image, blur[:len(image.shape)], preserve_range=True, multichannel=False)
    return image

def covert_tforms(tform):
    tforms_xyz = {k: (v[0][0], v[0][1], int(np.round(np.mean(v[0][2])))) for k, v in tform.items() if k!='nucstain'}
    txy = {k: (v[0], v[1]) for k, v in tforms_xyz.items()}
    tzz = {k: v[2] for k, v in tforms_xyz.items()}
    return txy,tzz

        
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
    tforms_path = args.tforms_path
    cword_config = args.cword_config
    global_kernel = (args.kernel_sigma,args.kernel_sigma,0)
    global_blur = (args.blur_sigma,args.blur_sigma,0)
    psf_map = {'Orange': orange_psf, 'FarRed': farred_psf, 'Green': green_psf}
    seqfish_config = importlib.import_module(cword_config)
    nbits = seqfish_config.nbits
    print(args)
    
    if use_gpu != 0:
        if ncpu>1:
            raise ValueError('Cannot use GPU if nCPU >1')
        print(device_lib.list_local_devices())
        gpu_algorithm = fd_restoration.RichardsonLucyDeconvolver(2).initialize()
        #assert(ncpu==1, 'If using GPU only use single-threaded to prevent conflicts with GPU usage.')
    else:
        gpu_algorithm = None
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    md = Metadata(md_path)
    bitmap = seqfish_config.bitmap
    pfunc = partial(hdata_multi_z_pseudo_maxprjZ_wrapper, gpu_algo=gpu_algorithm, md=md, k=k, zstart=zstart, zskip=zskip,
                zmax=zmax, ndecon_iter=niter, bitmap=bitmap)
    good_positions = pickle.load(open(tforms_path, 'rb'))['good']
    func_inputs = []
    for pos,tform in good_positions.items():
        processing_path = os.path.join(out_path,pos,'processing.pkl')
        if os.path.exists(processing_path):
            processing = pickle.load(open(processing_path,'rb'))
            if len(processing)==nbits:
                continue
        txy,tzz = covert_tforms(tform)
        hdata = HybeData(os.path.join(out_path, pos))
        if use_gpu!=0 or ncpu==1:
            hdata_multi_z_pseudo_maxprjZ_wrapper(hdata, pos, txy, tzz, md, bitmap,
                                                 zstart=zstart, k=k, zskip=zskip,
                                                 zmax=zmax, ndecon_iter=niter)
        else:
            func_inputs.append((HybeData(os.path.join(out_path, pos)), pos, txy, tzz))
        
    if use_gpu==0 and ncpu>1:
        pfunc = partial(hdata_multi_z_pseudo_maxprjZ_wrapper, md=md, k=k, zstart=zstart, zskip=zskip, zmax=zmax, ndecon_iter=niter)
        with multiprocessing.Pool(ncpu) as ppool:
            sys.stdout.flush()
            ppool.starmap(pfunc, func_inputs)
            sys.stdout.flush()
