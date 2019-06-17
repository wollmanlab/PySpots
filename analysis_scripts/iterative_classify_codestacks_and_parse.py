#!/usr/bin/env python

import os
import sys
import pickle
import argparse
import importlib
import traceback
import numpy as np
import pandas as pd
import dill as pickle
import multiprocessing
from skimage import io
from metadata import Metadata
from functools import partial
from fish_results import HybeData
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter
from skimage.measure import regionprops, label
from scipy.spatial import distance_matrix,KDTree

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cstk_path", type=str, help="Path to folder containing codestack npz files.")
    parser.add_argument("cword_config", type=str, help="Path to python file initializing the codewords and providing bitmap variable.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=8, action='store', help="Number of cores to utilize (default 8x4MKL Threads).")
    parser.add_argument("-r", "--nrandom", type=int, dest="nrandom", default=50, action='store', help="Number of random positions to choose to fit norm_factor.")
    parser.add_argument("-n", "--niter", type=int, dest="niter", default=10, action='store', help="Number of iterations to perform.")
    parser.add_argument("-d", "--cword_dist", type=float, dest="cword_dist", default=0.5176, action='store', help="Threshold for distance between pixel and codeword for classification.")
    parser.add_argument("-c", "--classify", type=float, dest="classify", default=0, action='store', help="Do you want to run full classification on all positions with real and blank barcodes after iterative classification? (0 or 1).")
    parser.add_argument("-f", "--fresh", type=float, dest="fresh", default=0, action='store', help="Do you want to calculate 95th percentile for the codestacks? (0 or 1).")
    args = parser.parse_args()
    
def parse_classification_image(class_img, cstk, cvectors, genes, zindex, pix_thresh=0, ave_thresh=0):
    #class_imgs = data['class_img']
    #cstk = data['cstk']
    label2d = label((class_img+1).astype('uint16'), neighbors=8)
    properties = regionprops(label2d, (class_img+1).astype('uint16'))
    areas = []
    nclasses = []
#     df_rows = []
    multiclass_sets = 0
    #bit_values = defaultdict(list)
    gene_call_rows = []
    below_threshold_rows = []
    for prop in properties:
        coords = prop.coords
        centroid = prop.centroid
        classes = list(set(prop.intensity_image.flatten())-set([0]))
        #classes = list(set(list())-set([0]))
        if len(classes)==0:
            print('Label with no classes.', end='')
            pdb.set_trace()
            continue
        elif not len(classes)==1:
            #print('Labels need to be broken apart more than one classification found per label.', end='')
            #print(classes)
            pdb.set_trace()
            multiclass_sets+=1
            continue
        else:
            nclasses.append(len(classes))
            areas.append(prop.area)
        codeword_idx = classes[0]-1
        bits = np.where(cvectors[codeword_idx]>0)[0]
        #spot_pixel_values = defaultdict(list)
        spot_pixel_means = []
        spot_sums = 0
        for x, y in coords:
            cur_vals = cstk[x, y, bits]
            spot_pixel_means.append(cur_vals)
            for idx, b in enumerate(bits):
                #spot_pixel_values[b].append(cur_vals[idx])
                #bit_values[b].append(cur_vals[idx])
                spot_sums += cur_vals[idx]
        if (len(coords)>pix_thresh) and (np.mean(spot_pixel_means)>ave_thresh):
            gene_call_rows.append([genes[codeword_idx], spot_sums, centroid,
                            np.mean(spot_pixel_means), len(coords), codeword_idx])
        else:
            below_threshold_rows.append([genes[codeword_idx], spot_sums, centroid,
                        np.mean(spot_pixel_means), len(coords), codeword_idx])
    df = pd.DataFrame(gene_call_rows, columns=['gene', 'ssum', 'centroid', 'ave', 'npixels', 'cword_idx'])
    return df

def multi_z_class_parse_wrapper(hdata, cvectors, genes, return_df = False):
#     data = np.load(f)
#     cstks, nfs, class_imgs = data['cstks'].tolist(), data['norm_factors'].tolist(), data['class_imgs'].tolist()
    pos = hdata.posname
    print(pos)
    cvectors = cvectors.copy()
    np.place(cvectors, cvectors>0, 1.)
#     data.close()
    merged_df =[]
    for z in hdata.metadata.zindex.unique():
        cstk = hdata.load_data(pos, z, 'cstk')
        class_img = hdata.load_data(pos, z, 'cimg')
        df = parse_classification_image(class_img, cstk, cvectors, genes, z)
        df['z'] = z
        df['posname'] = pos
        hdata.add_and_save_data(df,pos,z,'spotcalls')
#         merged_df.append(df)
#     if len(merged_df)>0:
#         merged_df = pd.concat(merged_df, ignore_index=True)
#         pickle.dump(merged_df, open(os.path.join(hdata.base_path, 'spotcalls.pkl'), 'wb'))
#         if return_df:
#             return merged_df
#     else:
#         if return_df:
    return None

def find_bitwise_error_rate(df, cvectors, norm_factor):
    cvectors = cvectors.copy()
    np.place(cvectors, cvectors>0., 1.)
    error_counts = Counter()
    bit_freq = Counter()
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(idx)
        cword = cvectors[row.cword_idx]
        cidx = np.where(cword==1.)[0]
        cnf = norm_factor[cidx]#/1.5
        bit_freq.update(cidx)
        bmeans = np.mean(row.pixel_values, axis=0)
        norm_bmeans = np.divide(bmeans, cnf)
        dists = distance_matrix(norm_bmeans[np.newaxis, :], [[1, 1, 1, 1],
                                                 [0, 1, 1, 1], 
                                                 [1, 0, 1, 1],
                                                 [1, 1, 0, 1],
                                                 [1, 1, 1, 0]])
        min_dist = np.argmin(dists)
        if min_dist == 0:
            continue
        else:
            berror = cidx[min_dist-1]
            error_counts[berror]+=1
        #print(berror)
        #break
    return error_counts, bit_freq


def purge_zoverlap(df, z_dist = 2):
    zidxes = df.z.unique()
    for z_i in range(len(zidxes)-1):
        subdf = df[(df.z==zidxes[z_i]) | (df.z==zidxes[z_i+1])]
        yx = subdf.centroid.values
        yx = np.stack(yx, axis=0)
        tree = KDTree(yx)
        dclust = DBSCAN(eps=2, min_samples=2)
        dclust.fit(yx)
        skip_list = set(np.where(dclust.labels_==-1)[0])
        nomatches = []
        drop_list = []
        for idx, i in enumerate(yx):
            if idx % 10000 == 0:
                print(idx)
            if idx in skip_list:
                continue
            m = tree.query_ball_point(i, 2)
            m = [j for j in m if j!=idx]

            row_query = subdf.iloc[idx]
            for j in m:
                row_match = subdf.iloc[j]
                if row_match.cword_idx!=row_query.cword_idx:
                    continue

                if row_match.npixels>=row_query.npixels:
                    drop_list.append((idx, j))
                else:
                    drop_list.append((j, idx))
                    break
        if len(drop_list)>0:
            droppers, keepers = zip(*drop_list)
            df.drop(index=subdf.iloc[list(droppers)].index,inplace=True)
    return df

def purge_zoverlap_pool(inputs,z_dist = 2):
    pos = inputs['posname']
    print('Starting ',pos)
    posdf = inputs['posdf']
    zidxes = posdf.z.unique()
    posdic = {}
    for z_i in range(len(zidxes)-1):
        subdf = posdf[(posdf.z==zidxes[z_i]) | (posdf.z==zidxes[z_i+1])]
        yx = subdf.centroid.values
        yx = np.stack(yx, axis=0)
        tree = KDTree(yx)
        dclust = DBSCAN(eps=2, min_samples=2)
        dclust.fit(yx)
        skip_list = set(np.where(dclust.labels_==-1)[0])
        nomatches = []
        drop_list = []
        for idx, i in enumerate(yx):
            if idx in skip_list:
                continue
            m = tree.query_ball_point(i, 2)
            m = [j for j in m if j!=idx]

            row_query = subdf.iloc[idx]
            for j in m:
                row_match = subdf.iloc[j]
                if row_match.cword_idx!=row_query.cword_idx:
                    continue

                if row_match.npixels>=row_query.npixels:
                    drop_list.append((idx, j))
                else:
                    drop_list.append((j, idx))
                    break
        if len(drop_list)>0:
            droppers, keepers = zip(*drop_list)
            index=subdf.iloc[list(droppers)].index
            posdic[z_i] = index
    return posdic,pos
                             
def purge_wrapper(df,ncpu):
    poses = df.posname.unique()
    dfdic = {}
    with multiprocessing.Pool(ncpu) as ppool:
        for result,pos in ppool.imap(purge_zoverlap_pool, [{'posname': pos, 'posdf': df[df.posname==pos]} for pos in poses]):
            dfdic[pos] = result
            print(pos,' Finished')
        drop_list = []
        for pos in dfdic.keys():
            for z in dfdic[pos].keys():
                droppers = dfdic[pos][z]
                for spot in droppers:
                    drop_list.append(spot)
        purged_df = df.drop(index = list(dict.fromkeys(drop_list)),inplace=False)
    return purged_df

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

def classify_codestack(cstk, norm_vector, codeword_vectors, csphere_radius=0.5176, intensity=300):
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
    np.place(cstk, cstk<=0, 0.01)
    # Normalize for intensity difference between codebits
    normstk = np.divide(cstk, norm_vector)
    # Prevent possible underflow/divide_zero errors
    # Fill class img one column at a time
    class_img = np.empty((cstk.shape[0], cstk.shape[1]), dtype=np.int16)
    for i in range(cstk.shape[0]):
        v = normstk[i, :, :]  # v is shape (2048, n_bits)
        # l2 norm note codeword_vectors should be prenormalized
        v = normalize(v, norm='l2')
        # Distance from unit vector of codewords and candidate pixel codebits
        d = distance_matrix(codeword_vectors, v)
        # Check if distance to closest unit codevector is less than csphere thresh
        dmin = np.argmin(d, axis=0)
        dvs = np.array([i if d[i, idx]<csphere_radius else -1 for idx, i in enumerate(dmin)])
        temp_cvs = codeword_vectors[dvs]
        temp_cvs[temp_cvs>0] = 1
        n_bits = np.sum(temp_cvs[0])
        ssums = np.sum(cstk[i,:,:]*temp_cvs, axis=1) 
        dvs[ssums<intensity*n_bits] = -1 # multiplying thresh by n_bits to scale it back up because averaging ssums is inefficient
        class_img[i, :] = dvs
    return class_img

def mean_one_bits(cstk, class_img, cvectors, spot_thresh=10**2.55,mean_min_spot_thresh = 485):#, nbits = 18):
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
    mean_background = []
    for i in range(cvectors.shape[0]):
        x, y = np.where(class_img==i)
        if len(x) == 0:
            continue
        onebits = np.where(cvectors[i,:]==1.)[0]
        if len(onebits)<1:
            continue
        for i in onebits:
            if np.mean(cstk[x, y, i])<mean_min_spot_thresh:
                mean_background.append(np.mean(cstk[x, y, i]))
                continue
            bitvalues[i].append(np.mean(cstk[x, y, i]))
    mean_bits = []
    for i in range(nbits):
        bvals = bitvalues[i]
        if len(bvals) == 0:
            mean_bits.append(np.nan)
        else:
            mean_bits.append(robust_mean(bitvalues[i]))
    return np.array(mean_bits),mean_background

def egalitarian_mean_one_bits(cstk, class_img, cvectors, spot_thresh=10**2.55):
    cstk = cstk.astype('float32')
    nf = np.zeros(cvectors.shape[1])
    #weights = np.zeros(cvectors.shape[1])
    for bit in range(cvectors.shape[1]):
        spots_intensity = []
        spotsum = 0
        for gene in range(cvectors.shape[0]):
            if cvectors[gene,bit]>0:
                spotsum = spotsum + len(class_img[class_img==gene])
                x, y = np.where(class_img==gene)
                if len(x) == 0:
                    continue
                intensities = cstk[x, y, bit]
                spots_intensity.append(intensities[intensities>spot_thresh])
        try:

            nf[bit]=np.mean(np.concatenate(spots_intensity).ravel())
        except:

            nf[bit] = np.nan
    return nf



def robust_mean(x):
    return np.average(x, weights=np.ones_like(x) / len(x))

def classify_file(hdata, nfactor, nvectors, genes, genesubset=None, intensity=300, csphere_radius=0.5176):
    """
    Wrapper for classify_codestack. Can change this instead of function if 
    intermediate file storage ever changes.
    """
    pos = hdata.posname
    cvectors = nvectors.copy()
    np.place(cvectors, cvectors>0., 1.) # Hack to allow normalized vectors as single input
    print(pos)
    zindexes = hdata.metadata.zindex.unique()
    nfs = {}
    background_intensity = []
    for z in zindexes:
        cstk = hdata.load_data(pos, z, 'cstk')
        new_class_img = classify_codestack(cstk, nfactor, nvectors, csphere_radius=csphere_radius,intensity=intensity)
        df = parse_classification_image(new_class_img, cstk, nvectors, genes, z)
        df['z'] = z
        df['posname'] = pos
        if genesubset is None:
            new_nf = egalitarian_mean_one_bits(cstk, new_class_img, cvectors)
        else:
            new_nf = egalitarian_mean_one_bits(cstk, new_class_img, cvectors[genesubset, :])

        hdata.add_and_save_data(new_nf, pos, z, 'nf')
        hdata.add_and_save_data(new_class_img, pos, z, 'cimg')
        hdata.add_and_save_data(df,pos,z,'spotcalls')

def unix_find(pathin):
    """Return results similar to the Unix find command run without options
    i.e. traverse a directory tree and return all the file paths
    """
    
    return [os.path.join(path, file)
            for (path, dirs, files) in os.walk(pathin)
            for file in files]
    

def calc_new_nf(hdata,percentile=95):
    pos = hdata.posname
    for z in hdata.metadata.zindex.unique():
        try:
            cstk = hdata.load_data(pos,z,'cstk')
            nf = np.percentile(cstk, percentile, axis=(0, 1))
            hdata.add_and_save_data(nf,pos,z,'nf')
        except:
            print(pos,z,'Failed to calc new nf')
            print(nf)
    return None

if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['GOTO_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    print(args)
    cstk_path = args.cstk_path
    ncpu = args.ncpu
    niter = args.niter
    nrandom = args.nrandom
    cword_radius = args.cword_dist
    # Assuming these get imported during call below:
    # 1. bitmap
    # 2. bids, blanks, gids, cwords, gene_codeword_vectors, blank_codeword_vectors
    # 3. norm_gene_codeword_vectors, norm_blank_codeword_vectors
    seqfish_config = importlib.import_module(args.cword_config)
    try:
        genes = seqfish_config.gids+seqfish_config.bids
    except:
        genes = seqfish_config.gids
    bitmap = seqfish_config.bitmap
    normalized_gene_vectors = seqfish_config.norm_gene_codeword_vectors
    normalized_all_gene_vectors = seqfish_config.norm_all_codeword_vectors
    
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
    if niter > 0:
        if args.fresh != 0:
            # if redoing an iterative clasify you can calculate 95th percentile nf here
            print('Calculating 95th Percentile nf')
            with multiprocessing.Pool(ncpu) as ppool:
                sys.stdout.flush()
                results = ppool.map(calc_new_nf, hybedatas)
                ppool.close()
                sys.stdout.flush()
        for i in range(niter):
            print('Number of Positions: ', len(hybedatas))
            print('Iteration number ', i+1)
            
            #calculate cumulative normalization factors
            cur_nf = np.array(np.nanmean([mean_nfs(hdata) for hdata in hybedatas], axis=0))
            if i>0:
                cur_nf = np.array([10**2.6 if (i<10**2.6) or np.isnan(i) else i for i in cur_nf])
            print(cur_nf)
            # itterativly classify each position in pools of ncpu
            with multiprocessing.Pool(ncpu) as ppool:
                sys.stdout.flush()
                classify_pfunc = partial(classify_file, nfactor=cur_nf, nvectors=normalized_gene_vectors, genes=genes, csphere_radius=cword_radius)
                results = ppool.map(classify_pfunc, hybedatas)
                ppool.close()
                sys.stdout.flush()

    # After all of your rounds of itterativly classifying now use those nf to classify all positions
    if args.classify != 0:
        print('Starting final classification using all barcodes and all positions')
        cur_nf = np.array(np.nanmean([mean_nfs(hdata) for hdata in hybedatas], axis=0))
        cur_nf = np.array([10**2.6 if (i<10**2.6) or np.isnan(i) else i for i in cur_nf])
        print(cur_nf)
        poses = [i for i in os.listdir(cstk_path) if os.path.isdir(os.path.join(cstk_path, i))]
        hybedatas = [HybeData(os.path.join(cstk_path, i)) for i in poses]
        with multiprocessing.Pool(ncpu) as ppool:
            sys.stdout.flush()
            classify_pfunc = partial(classify_file, nfactor=cur_nf, nvectors=normalized_gene_vectors, genes=genes, csphere_radius=cword_radius)
            results = ppool.map(classify_pfunc, hybedatas)
            ppool.close()
            sys.stdout.flush()

