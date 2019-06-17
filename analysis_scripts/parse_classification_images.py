from skimage.measure import regionprops, label
import pandas as pd
import numpy as np
from fish_results import HybeData
from collections import defaultdict, Counter
from scipy.spatial import distance_matrix
from metadata import Metadata
from functools import partial
import importlib
import multiprocessing
import pickle
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cstk_path", type=str, help="Path to folder containing codestack npz files.")
    parser.add_argument("cword_config", type=str, help="Path to python file initializing the codewords and providing bitmap variable.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=64, action='store', help="Number of cores to utilize (default 8x4MKL Threads).")
#     parser.add_argument("-c", "--coords", type=int, dest="coords", default=0, action='store', help="Do you want to add position coordinate data to df? (0,1)")
#     parser.add_argument("-m", "--md_path", type=str, dest="md_path", default=False, action='store', help="Metadata Path for finding position coordinates")
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
    bit_values = defaultdict(list)
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
        spot_pixel_values = defaultdict(list)
        spot_pixel_means = []
        spot_sums = 0
        for x, y in coords:
            cur_vals = cstk[x, y, bits]
            spot_pixel_means.append(cur_vals)
            for idx, b in enumerate(bits):
                spot_pixel_values[b].append(cur_vals[idx])
                bit_values[b].append(cur_vals[idx])
                spot_sums += cur_vals[idx]
        if (len(coords)>pix_thresh) and (np.mean(spot_pixel_means)>ave_thresh):
            gene_call_rows.append([genes[codeword_idx], spot_sums, centroid, spot_pixel_values,
                            np.mean(spot_pixel_means), len(coords), codeword_idx, coords])
        else:
            below_threshold_rows.append([genes[codeword_idx], spot_sums, centroid, spot_pixel_values,
                        np.mean(spot_pixel_means), len(coords), codeword_idx, coords])
    df = pd.DataFrame(gene_call_rows, columns=['gene', 'ssum', 'centroid', 'pixel_values', 'ave', 'npixels', 'cword_idx', 'coords'])
    return df, bit_values
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
        df, bvs = parse_classification_image(class_img, cstk, cvectors, genes, z)
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

from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

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

if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['GOTO_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    print(args)
    cstk_path = args.cstk_path
    ncpu = args.ncpu
    seqfish_config = importlib.import_module(args.cword_config)
    cvectors = seqfish_config.norm_all_codeword_vectors
    try:
        genes = seqfish_config.gids+seqfish_config.bids
    except:
        genes = seqfish_config.gids
    poses = [i for i in os.listdir(cstk_path) if os.path.isdir(os.path.join(cstk_path, i))]
    hybedatas = [HybeData(os.path.join(cstk_path, i)) for i in poses]
    
    with multiprocessing.Pool(ncpu) as ppool:
        parse_pfunc = partial(multi_z_class_parse_wrapper,cvectors=cvectors,genes=genes)
        results = ppool.map(parse_pfunc, hybedatas)
    print('Combining all df')
    spotcalls = []
    for hdata in hybedatas:
            for zindex in hdata.metadata.zindex.unique():
                try:
                    spotcalls.append(hdata.load_data(hdata.posname,zindex,'spotcalls'))
                except:
                    continue
    spotcalls = pd.concat(spotcalls,ignore_index=True)
    pickle.dump(spotcalls,open(os.path.join(cstk_path,'spotcalls.csv'),'wb'))

    print('Finished')