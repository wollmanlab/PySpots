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
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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
    parser.add_argument("-f", "--fresh", type=float, dest="fresh", default=0, action='store', help="Do you want to calculate 95th percentile for the codestacks? (0 or the percentile).")
    parser.add_argument("-zp", "--purge", type=float, dest="purge", default=0, action='store', help="Do you want to purge and concatenate spotcalls? (0 or 1).")
    parser.add_argument("-s", "--system", type=str, dest="system", default='None', action='store', help="What dataset do you want to check correlation with?")
    args = parser.parse_args()
    
def parse_classification_image(class_img, cstk, cvectors, genes, zindex, pix_thresh=1, ave_thresh=0,spot_sum_thresh=0):
    #ave_thresh=1000,spot_sum_thresh=2**13
    # Calculating nf during parse for better control
    # Set up a dictionary where each key is a bit
    # Each bit will have a dictionary of spots
    # Each Spot dict will have the mean spot intensities for all spots 
    #     classified above spot_sum_thresh
    nf_dict = {}
    for bit in range(cvectors.shape[1]):
        nf_dict[bit] = {}
        for gene in genes:
            nf_dict[bit][gene]=[]
    label2d = label((class_img+1).astype('uint16'), neighbors=8)
    properties = regionprops(label2d, (class_img+1).astype('uint16'))
    areas = []
    nclasses = []
    multiclass_sets = 0
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
            pdb.set_trace()
            multiclass_sets+=1
            continue
        else:
            nclasses.append(len(classes))
            areas.append(prop.area)
        codeword_idx = classes[0]-1
        gene = genes[codeword_idx]
        bits = np.where(cvectors[codeword_idx]>0)[0]
        spot_pixel_values = []
        spot_pixel_means = []
        spot_sums = 0
        # Calculating the mean pixel intensities for each positive bit for a single spot
        spot_nf = np.zeros(cvectors.shape[1])
        for b in bits:
            spot_bit_intensities = cstk[coords[:,0], coords[:,1], b]
            spot_nf[b] = np.mean(spot_bit_intensities)
            spot_pixel_values.append(spot_bit_intensities)
        spot_sum = np.sum(spot_pixel_values)
        spot_mean = np.mean(spot_pixel_values)
        # If the spot is above spot_sum_thresh then add it to the gene spot list
        # the hope is to filter out background here
        if (len(coords)>pix_thresh) and (spot_mean>ave_thresh) and (spot_sum>spot_sum_thresh):
            for b in bits:
                nf_dict[b][gene].append(spot_nf[b])
        # if a spot is above these thresholds add it to the final data frame
#         if (len(coords)>pix_thresh) and (spot_mean>ave_thresh) and (spot_sum>spot_sum_thresh):
        gene_call_rows.append([genes[codeword_idx], spot_sum, centroid,
                        np.mean(spot_mean), len(coords), codeword_idx])
#         else:
#             below_threshold_rows.append([genes[codeword_idx], spot_sum, centroid,
#                         np.mean(spot_mean), len(coords), codeword_idx])
    df = pd.DataFrame(gene_call_rows, columns=['gene', 'ssum', 'centroid', 'ave', 'npixels', 'cword_idx'])
    # Generate an empty nf to populate
    nf = np.zeros(cvectors.shape[1])
#     if np.mean(df.ave)<1000: # Arbitrary thresh between real and fake
#         nf[nf==0]=np.nan
        # Prevents bad postions from effecting normalization factors
#     else:
    for b in nf_dict.keys():
        bit_nf = []
        # mean the spot intensities for a gene so that all genes have the same weight
        # This is to prevent highly expressed false spots from skewing nf
        for gene,gene_bit_spot_intensities in nf_dict[b].items():
            if len(gene_bit_spot_intensities)>0:
                bit_nf.append(np.nanmean(gene_bit_spot_intensities))
        nf[b] = np.nanmean(bit_nf)
    return df,nf

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
    pos = df.posname.iloc[0]
    print('Starting ',pos)
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
#             if idx % 10000 == 0:
#                 print(idx)
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
                             
def purge_wrapper(df,ncpu):
    poses = df.posname.unique()
    pos_dfs = []
    for pos in poses:
        pos_dfs.append(df[df.posname==pos])
    with multiprocessing.Pool(ncpu) as ppool:
        purged_list = []
        for result in ppool.imap(purge_zoverlap, pos_dfs):
            purged_list.append(result)
            print(result.posname.iloc[0],' Finished')
    purged_df = pd.concat(purged_list,ignore_index=True)
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
            z = int(z)
        except:
            continue
        nf = hdata.load_data(pos, z, 'nf')
        nfs.append(nf)
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

def classify_codestack(cstk, mask, norm_vector, codeword_vectors, csphere_radius=0.5176, intensity=500):
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
    ptiles = np.percentile(cstk, 75, axis=(0,1))
    cstk = np.subtract(cstk, ptiles[np.newaxis, np.newaxis, :])
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
        v[mask[i,:]==False]=False
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

def mean_one_bits(cstk, class_img, cvectors, spot_thresh=10**2.55,mean_min_spot_thresh = 485):
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
    return np.array(mean_bits)

def egalitarian_mean_one_bits(cstk, class_img, cvectors, spot_thresh=10**2.55,min_spot_count=10):
    cstk = cstk.astype('float32')
    nf = np.zeros(cvectors.shape[1])

    for bit in range(cvectors.shape[1]):
        spots_intensity = []
        spotsum = 0
        for gene in range(cvectors.shape[0]):
            if cvectors[gene,bit]>0:
                spotsum = spotsum + len(class_img[class_img==gene])
                x, y = np.where(class_img==gene)
                if len(x) < min_spot_count:
                    continue
                intensities = cstk[x, y, bit]
                intensities = intensities[intensities>spot_thresh]
                if len(intensities>0):
                    spots_intensity.append(np.mean(intensities))
        if len(spots_intensity)>0:
            nf[bit]=np.mean(spots_intensity)
        else:
            nf[bit] = np.nan
    return nf



def robust_mean(x):
    return np.average(x, weights=np.ones_like(x) / len(x))

def classify_file(hdata, nfactor, nvectors, genes, genesubset=None, intensity=0, csphere_radius=0.5176,mask_img=False):
    """
    Wrapper for classify_codestack. Can change this instead of function if 
    intermediate file storage ever changes.
    """
    processing = pickle.load(open(os.path.join(hdata.base_path,'processing.pkl'),'rb'))
    pos = hdata.posname
    if len(processing)==18:
        cvectors = nvectors.copy()
        np.place(cvectors, cvectors>0., 1.) # Hack to allow normalized vectors as single input
        #print(pos)
        nfs = {}
        background_intensity = []
        spotcall_list = []
        for z in hdata.metadata.zindex.unique():
            try:
                z = int(z)
            except:
                continue
            cstk = hdata.load_data(pos, z, 'cstk')
            if mask_img==True:
                mask = hdata.load_data(pos, z, 'cstk')
            else:
                mask = np.zeros_like(cstk[:,:,0])
                mask = mask>-1
   