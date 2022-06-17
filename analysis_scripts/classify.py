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
    parser.add_argument("-f", "--fresh", type=float, dest="fresh", default=0, action='store', help="Do you want to calculate percentile for the codestacks? (0(No) or the percentile).")
    parser.add_argument("-zp", "--purge", type=float, dest="purge", default=0, action='store', help="Do you want to purge and concatenate spotcalls? (0 or 1).")
    args = parser.parse_args()
    
def parse_classification_image(class_img, cstk, cvectors, genes, zindex, distance_img, pix_thresh=1, ave_thresh=10**1.3, spot_sum_thresh=10**2.5,spot_dist_thresh=0.45):
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
    label2d = label((class_img+1).astype('uint16'), connectivity=2)
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
        if len(classes)==0:
            print('Label with no classes.')
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
        # Calculating the mean pixel intensities for each positive bit for a single spot
        spot_nf = np.zeros(cvectors.shape[1])
        for b in bits:
            spot_bit_intensities = cstk[coords[:,0], coords[:,1], b]
            spot_nf[b] = np.mean(spot_bit_intensities)
            spot_pixel_values.append(spot_bit_intensities)
        spot_sum = np.sum(spot_pixel_values)
        spot_mean = np.mean(spot_pixel_values)
        spot_distance = np.mean(distance_img[coords[:,0], coords[:,1]])
        # If the spot is above spot_sum_thresh then add it to the gene spot list
        # the hope is to filter out background here
        if (len(coords)>pix_thresh) and (spot_mean>ave_thresh) and (spot_sum>spot_sum_thresh) and (spot_distance<spot_dist_thresh):
            for b in bits:
                nf_dict[b][gene].append(spot_nf[b])
        gene_call_rows.append([genes[codeword_idx], spot_sum, centroid,
                        spot_mean, len(coords), codeword_idx,spot_distance])
    df = pd.DataFrame(gene_call_rows, columns=['gene', 'ssum', 'centroid', 'ave', 'npixels', 'cword_idx','cword_dist'])
    # Generate an empty nf to populate
    nf = np.zeros(cvectors.shape[1])
    for b in nf_dict.keys():
        bit_nf = []
        # mean the spot intensities for a gene so that all genes have the same weight
        # This is to prevent highly expressed false spots from skewing nf
        for gene,gene_bit_spot_intensities in nf_dict[b].items():
            if len(gene_bit_spot_intensities)>0:
                bit_nf.append(np.mean(gene_bit_spot_intensities))
        nf[b] = np.mean(bit_nf)
    return df,nf

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
    return error_counts, bit_freq

def purge_zoverlap(df, z_dist = 2):
    pos = df.posname.iloc[0]
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
    np.place(cstk, cstk<=0, 0.01)
    # Normalize for intensity difference between codebits
    normstk = np.divide(cstk, norm_vector)
    # Prevent possible underflow/divide_zero errors
    # Fill class img one column at a time
    class_img = np.empty((cstk.shape[0], cstk.shape[1]), dtype=np.int16)
    distance_img = np.empty((cstk.shape[0], cstk.shape[1]), dtype=np.float64)
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
        class_img[i, :] = dvs
        distance_img[i,:] = np.min(d,axis=0)
    return class_img,distance_img

def classify_file(hdata, nfactor, nvectors, genes, genesubset=None, intensity=0, csphere_radius=0.5176,mask_img=False):
    """
    Wrapper for classify_codestack. Can change this instead of function if 
    intermediate file storage ever changes.
    """
    processing = pickle.load(open(os.path.join(hdata.base_path,'processing.pkl'),'rb'))
    pos = hdata.posname
    if len(processing)==18:
        cvectors = nvectors.copy()
        np.place(cvectors, cvectors>0., 1.)
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
            new_class_img,distance_img = classify_codestack(cstk,mask, nfactor, nvectors, csphere_radius=csphere_radius,intensity=intensity)
            df,new_nf = parse_classification_image(new_class_img, cstk, nvectors, genes, z,distance_img)
            df['z'] = z
            df['posname'] = pos
            spotcall_list.append(df)
            hdata.add_and_save_data(new_nf, pos, z, 'nf')
            hdata.add_and_save_data(new_class_img, pos, z, 'cimg')
            hdata.add_and_save_data(df,pos,z,'spotcalls')
        spotcalls = pd.concat(spotcall_list)
        hdata.add_and_save_data(spotcalls,pos,'all','spotcalls')
    else:
        print(pos,'cstk isnt complete')
        
def calc_new_nf(hdata,percentile=95):
    pos = hdata.posname
    for z in hdata.metadata.zindex.unique():
        try:
            z = int(z)
        except:
            continue
        cstk = hdata.load_data(pos,z,'cstk')
        nf = np.percentile(cstk, percentile, axis=(0, 1))
        if len(nf)>0:
            hdata.add_and_save_data(nf,pos,z,'nf')
        else:
            print('nf len is 0')
            print(pos,z)
            nf = np.zeros(cstk.shape[2])
            nf[nf==0]=float('NaN')
            hdata.add_and_save_data(nf,pos,z,'nf')
    return None

def find_finished_pos(poses,cstk_path,nbits):
    finished_pos = []
    for pos in poses:
        try:
            processing = pickle.load(open(os.path.join(cstk_path,pos,'processing.pkl'),'rb'))
            if len(processing)==nbits: # Check to make sure position is finished
                finished_pos.append(pos)
        except:
            continue
    return finished_pos        
    
def random_subset(nrandom,finished_pos):
    if nrandom<len(finished_pos):
        subset = np.random.choice(finished_pos, size=nrandom, replace=False)
        return subset 
    else:
        np.random.shuffle(finished_pos)
        return finished_pos

def find_subset(cstk_path,nrandom,nbits):
    # Load Positions and Generate Random Subset
    poses = [i for i in os.listdir(cstk_path) if os.path.isdir(os.path.join(cstk_path, i))]
    finished_pos = find_finished_pos(poses,cstk_path,nbits)
    subset = random_subset(nrandom,finished_pos)
    return subset,poses
        
def spotcat(hybedatas):
    spotcalls = []
    for hdata in hybedatas:
        for zindex in hdata.metadata.zindex.unique():
            if ('hybe' in zindex) or ('all' in zindex):
                continue
            try:
                spotcalls.append(hdata.load_data(hdata.posname,zindex,'spotcalls'))
            except:
                print(hdata.posname,zindex,' failed to load spotcalls')
                continue
    spotcalls = pd.concat(spotcalls,ignore_index=True)
    return spotcalls
    
if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['GOTO_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    cstk_path = args.cstk_path
    ncpu = args.ncpu
    niter = args.niter
    nrandom = args.nrandom
    cword_radius = args.cword_dist
    classify = args.classify
    fresh = args.fresh
    purge = args.purge
    print(args)
    
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
    nbits = seqfish_config.nbits
    
    # Load Positions and Generate Random Subset
    subset,poses = find_subset(cstk_path,nrandom,nbits)
    pickle.dump(subset,open(os.path.join(cstk_path,'subset.pkl'),'wb'))
#     subset = pickle.load(open(os.path.join(cstk_path,'subset.pkl'),'rb'))
    print(subset)
    hybedatas = [HybeData(os.path.join(cstk_path, i)) for i in subset]
    
    # if redoing an iterative clasify you can calculate 95th percentile nf here
    if fresh != 0:
            print('Calculating ',args.fresh,'th Percentile nf')
            with multiprocessing.Pool(ncpu) as ppool:
                sys.stdout.flush()
                calc_new_nf_pfunc = partial(calc_new_nf,percentile=fresh)
                results = ppool.map(calc_new_nf_pfunc, hybedatas)
                ppool.close()
                sys.stdout.flush()
                
    if niter > 0:
        for i in range(niter):
            print('Number of Positions: ', len(hybedatas))
            print('Iteration number ', i+1)
            
            #calculate cumulative normalization factors
            cur_nf = np.array(np.nanmean([mean_nfs(hdata) for hdata in hybedatas], axis=0))
            print(cur_nf)
            
            # itterativly classify each position in pools of ncpu
            with multiprocessing.Pool(ncpu) as ppool:
                sys.stdout.flush()
                classify_pfunc = partial(classify_file, nfactor=cur_nf, nvectors=normalized_gene_vectors, genes=genes, csphere_radius=cword_radius,mask_img=True)
                results = ppool.map(classify_pfunc, hybedatas)
                ppool.close()
                sys.stdout.flush()
            spotcalls = spotcat(hybedatas)
            pickle.dump(spotcalls,open(os.path.join(cstk_path,'spotcalls_iteration_'+str(i)+'.pkl'),'wb'))

    # After all of your rounds of itterativly classifying now use those nf to classify all positions
    if classify != 0: 
        print('Starting final classification using all barcodes and all positions')
        cur_nf = np.array(np.nanmean([mean_nfs(hdata) for hdata in hybedatas], axis=0))
        print(cur_nf)
        poses = [i for i in os.listdir(cstk_path) if os.path.isdir(os.path.join(cstk_path, i))]
        hybedatas = [HybeData(os.path.join(cstk_path, i)) for i in poses]
        with multiprocessing.Pool(ncpu) as ppool:
            sys.stdout.flush()
            classify_pfunc = partial(classify_file, nfactor=cur_nf, nvectors=normalized_all_gene_vectors, genes=genes, csphere_radius=cword_radius,mask_img=False)
            results = ppool.map(classify_pfunc, hybedatas)
            ppool.close()
            sys.stdout.flush()
        spotcalls = spotcat(hybedatas)
        pickle.dump(spotcalls,open(os.path.join(cstk_path,'spotcalls_prepurge.pkl'),'wb'))
    # After Classification Now Purge overlapping spots
    if purge != 0:
        purged_df = purge_wrapper(spotcalls,ncpu)
        pickle.dump(purged_df,open(os.path.join(cstk_path,'spotcalls.pkl'),'wb'))