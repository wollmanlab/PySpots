#!/usr/bin/env python
from skimage import io
import os
import sys
from collections import defaultdict
import numpy
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance_matrix
from skimage.morphology import remove_small_objects
from functools import partial
import importlib
import multiprocessing
import dill as pickle
import traceback
from fish_results import HybeData

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cstk_path", type=str, help="Path to folder containing codestack npz files.")
    parser.add_argument("cword_config", type=str, help="Path to python file initializing the codewords and providing bitmap variable.")
#     parser.add_argument("--posnames", type=str, help="Path pickle dictionary of tforms per position name.")
    #parser.add_argument("out_path", type=str, help="Path to save output.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=8, action='store', help="Number of cores to utilize (default 8x4MKL Threads).")
#     parser.add_argument("--posnames", dest="posnames", nargs='*', type=str, default=[], action='store', help="Number z-slices above and below to max project together.")
    parser.add_argument("-r", "--nrandom", type=int, dest="nrandom", default=50, action='store', help="Number of random positions to choose to fit norm_factor.")
    parser.add_argument("-n", "--niter", type=int, dest="niter", default=10, action='store', help="Number of iterations to perform.")
#     parser.add_argument("-m", "--zmax", type=int, dest="zmax", default=15, action='store', help="End making max projections centered at zmax.")
#     parser.add_argument("-i", "--zskip", type=int, dest="zskip", default=4, action='store', help="Skip this many z-slices between centers of max projections.")
    args = parser.parse_args()
    
def mean_nfs(hdata):
    """
    Iterate through codestacks and average norm factors.
    """
    pos = hdata.posname
    zindxes = hdata.metadata.zindex.unique()
    nfs = []
    for z in zindxes:
        try:
            nf = hdata.load_data(pos, z, 'nf')
            nfs.append(nf)
        except:
            print(pos)
            continue
    return np.nanmean(nfs, axis=0)

def cstk_mean_std(hdata):
    zindxes = hdata.metadata.zindex.unique()
    pos = hdata.posname
    if isinstance(pos, np.ndarray):
        if len(pos)==1:
            pos = pos[0]
        else:
            raise ValueError("More than one position found in this metadata.")
    elif not isinstance(pos, str):
        raise ValueError("unable load position name from HybeData metadata.")
    stds = []
    means = []
    for z in zindxes:
        cstk = hdata.load_data(pos, z, 'cstk')
        std = np.std(cstk, axis=(0,1))
        mean = np.mean(cstk, axis=(0,1))
        stds.append(std)
        means.append(mean)

#     means, stds = np.stack(means, axis=0), np.stack(stds, axis=0)
#     return np.nanmean(means, axis=0), np.nanmean(stds, axis=0)

def classify_codestack(cstk, norm_vector, codeword_vectors, csphere_radius=0.5176, clip_intensity=True):
    """
    Pixel based classification of codestack into gene_id pixels.
    
    Parameters
    ----------
    cstk : ndarray
        Codestack (y,x,codebit)
    norm_vector : array
        Array (ncodebits,1) used to normalization intensity variation between codebits.
    codeword_vectors : ndarray
        Vector Normalized Array (ncodewords, ncodebits) specifying one bits for each gene
    csphere_radius : float
        Radius of ndim-sphere serving as classification bubble around codeword vectors
    
    Returns
    -------
    class_img : ndarray
        Array (y,x) -1 if not a gene else mapped to index of codeword_vectors
    """
    cstk = cstk.copy()
    cstk = cstk.astype('float32')
    cstk = np.nan_to_num(cstk)
    if clip_intensity:
        cstk_means = np.mean(cstk, axis=(0, 1))
        for i in range(cstk.shape[2]):
            np.place(cstk[:,:,i], cstk[:,:,i]<=cstk_means[i], 0.01)
    else:
        np.place(cstk, cstk<=0., 0.01)
    # Normalize for intensity difference between codebits
    normstk = np.divide(cstk, norm_vector)
    # Prevent possible underflow/divide_zero errors
    # Fill class img one column at a time
    class_img = np.empty((cstk.shape[0], cstk.shape[1]), dtype=np.int16)
    for i in range(cstk.shape[0]):
        v = normstk[i, :, :]
        # l2 norm note codeword_vectors should be prenormalized
        v = normalize(v, norm='l2')
        # Distance from unit vector of codewords and candidate pixel codebits
        d = distance_matrix(codeword_vectors, v)
        # Check if distance to closest unit codevector is less than csphere thresh
        dmin = np.argmin(d, axis=0)
        dv = [i if d[i, idx]<csphere_radius else -1 for idx, i in enumerate(dmin)]
#         if not intensity is None:
#             # Enforce minimum intensity for pixel to be classified
#             pass
        class_img[i, :] = dv
#     remove_small = remove_small_objects((class_img+1).astype('bool'), min_size=2)
#     np.place(class_img, ~remove_small, -1)
    return class_img

def mean_one_bits(cstk, class_img, cvectors, spot_thresh=10**2.55):#, nbits = 18):
    """
    Calculate average intensity of classified pixels per codebits.
    
    Parameters
    ----------
    cstk : ndarray
        Codestack
    class_img : ndarray
        classification image
    nbits : int
        Number of codebits (maybe just make it equal to cstk shape?)
    
    Returns
    -------
    mean_bits : array
        Mean of classified pixels for each codebit
    """
    bitvalues = defaultdict(list)
    cstk = cstk.astype('float32')
    nbits = cstk.shape[2]
    for i in range(cvectors.shape[0]):
        x, y = np.where(class_img==i)
        if len(x) == 0:
            continue
        onebits = np.where(cvectors[i,:]==1.)[0]
        if len(onebits)<1:
            continue
        for i in onebits:
            bitvalues[i].append(np.mean(cstk[x, y, i]))
    mean_bits = []
    for i in range(nbits):
        bvals = bitvalues[i]
        if len(bvals) == 0:
            mean_bits.append(np.nan)
        else:
            mean_bits.append(robust_mean(bitvalues[i]))
    return np.array(mean_bits)

def robust_mean(x):
    return np.average(x, weights=np.ones_like(x) / len(x))
                    
def classify_file(hdata, nfactor, nvectors, genesubset=None, thresh=500):
    """
    Wrapper for classify_codestack. Can change this instead of function if 
    intermediate file storage ever changes.
    """
    pos = hdata.posname
    cvectors = nvectors.copy()
    np.place(cvectors, cvectors>0., 1.) # Hack to allow normalized vectors as single input
    print(pos)
    zindexes = hdata.metadata.zindex.unique()
#     pos = hdata.metadata.posname.unique()
    nfs = {}
    for z in zindexes:
#         try:
        cstk = hdata.load_data(pos, z, 'cstk')
        new_class_img = classify_codestack(cstk, nfactor, nvectors)
        #class_imgs[z] = new_class_img
        if genesubset is None:
            new_nf = mean_one_bits(cstk, new_class_img, cvectors)
        else:
            new_nf = mean_one_bits(cstk, new_class_img, cvectors[genesubset, :])
        hdata.add_and_save_data(new_nf, pos, z, 'nf')
        hdata.add_and_save_data(new_class_img, pos, z, 'cimg')
#         except:
#             hdata.remove_metadata_by_zindex(z)
#             continue

def unix_find(pathin):
    """Return results similar to the Unix find command run without options
    i.e. traverse a directory tree and return all the file paths
    """
    
    return [os.path.join(path, file)
            for (path, dirs, files) in os.walk(pathin)
            for file in files]
    
if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['GOTO_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    print(args)
    cstk_path = args.cstk_path
    ncpu = args.ncpu
    niter = args.niter
    nrandom = args.nrandom
    # Assuming these get imported during call below:
    # 1. bitmap
    # 2. bids, blanks, gids, cwords, gene_codeword_vectors, blank_codeword_vectors
    # 3. norm_gene_codeword_vectors, norm_blank_codeword_vectors
    seqfish_config = importlib.import_module(args.cword_config)
    bitmap = seqfish_config.bitmap
    normalized_gene_vectors = seqfish_config.norm_gene_codeword_vectors
    
    poses = [i for i in os.listdir(cstk_path) if os.path.isdir(os.path.join(cstk_path, i))]
    if nrandom<len(poses):
        subset = np.random.choice(poses, size=nrandom, replace=False)
    else:
        np.random.shuffle(poses)
        subset = poses
    print(subset)
    hybedatas = [HybeData(os.path.join(cstk_path, i)) for i in subset]

    
# Note preceding blocks and this can be noisy if restarted after crash etccc
    # Note preceding blocks and this can be noisy if restarted after crash etccc
    with multiprocessing.Pool(ncpu) as ppool:
        failed_positions = []
        for i in range(niter):
            print('N Positions left: ', len(hybedatas))
            if i == 0:
                cur_nf = np.array(np.nanmean([mean_nfs(hdata) for hdata in hybedatas], axis=0))
                print('90th Percentile Normalization factors:', cur_nf, sep='\n')
            else:
                cur_nf = np.array(np.nanmean([mean_nfs(hdata) for hdata in hybedatas], axis=0))
                cur_nf = np.array([10**2.6 if (i<10**2.6) or np.isnan(i) else i for i in cur_nf])
                print(cur_nf)
            sys.stdout.flush()
            classify_pfunc = partial(classify_file, nfactor=cur_nf, nvectors=normalized_gene_vectors)
            results = ppool.map(classify_pfunc, hybedatas)
