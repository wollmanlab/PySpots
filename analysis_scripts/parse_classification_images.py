from skimage.measure import regionprops, label
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial import distance_matrix
import pickle
import os

def parse_classification_image(class_img, cstk, cvectors, genes, zindex):
    #class_imgs = data['class_img']
    #cstk = data['cstk']
    label2d = label((class_img+1).astype('uint16'), neighbors=8)
    properties = regionprops(label2d, (class_img+1).astype('uint16'))
    areas = []
    nclasses = []
    df_rows = []
    multiclass_sets = 0
    bit_values = defaultdict(list)
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
            #different_z.append(len(set(coords[:,2])))
            nclasses.append(len(classes))
            areas.append(prop.area)
        codeword_idx = classes[0]-1
        bits = np.where(cvectors[codeword_idx]>0)[0]
#         spot_bit_values = {}
        spot_pixel_values = defaultdict(list)
        spot_pixel_means = []
        for x, y in coords:
            cur_vals = cstk[x, y, bits]
            spot_pixel_means.append(cur_vals)
            for idx, b in enumerate(bits):
                spot_pixel_values[b]+=list(cur_vals)
                bit_values[b].append(cur_vals[idx])
            #norm_spot_pixel_values = spot_pixel_values
            #norm_spot_pixel_values = nstk[x, y, bits][0]
            #pdb.set_trace()
        df_rows.append([genes[codeword_idx], centroid, spot_pixel_values,
                        np.mean(spot_pixel_means), len(coords), codeword_idx, coords])
    df = pd.DataFrame(df_rows, columns=['gene', 'centroid', 'pixel_values', 'ave', 'npixels', 'cword_idx', 'coords'])
    return df, bit_values
def multi_z_class_parse_wrapper(hdata, pos, cvectors, genes):
#     data = np.load(f)
#     cstks, nfs, class_imgs = data['cstks'].tolist(), data['norm_factors'].tolist(), data['class_imgs'].tolist()
    cvectors = cvectors.copy()
    np.place(cvectors, cvectors>0, 1.)
#     data.close()
    merged_df =[]
    for z in hdata.metadata.zindex.unique():
        cstk = hdata.load_data(pos, z, 'cstk')
        class_img = hdata.load_data(pos, z, 'cimg')
        df, bvs = parse_classification_image(class_img, cstk, cvectors, genes, z)
        df['z'] = z
        
        merged_df.append(df)
    merged_df = pd.concat(merged_df, ignore_index=True)
    merged_df['posname'] = pos
    pickle.dump(merged_df, open(os.path.join(hdata.base_path, 'spotcalls.pkl'), 'wb'))
    #return pd.concat(merged_df, ignore_index=True)

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


