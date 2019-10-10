#!/usr/bin/env python
import os
import scipy
import pickle
import numpy as np
import multiprocessing
from skimage import io
from itertools import repeat
from functools import partial
from metadata import Metadata
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from skimage.transform import resize
from skimage.filters import gaussian
from collections import Counter, defaultdict
from skimage.feature import match_template, peak_local_max

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str, help="Path to where your images are unless they are in Analysis/deconvolved/.",default = 'None')
    parser.add_argument("analysis_path", type=str, help="Path to where you want your data saved.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=6, action='store', help="Number of cores to utilize (default 6).")
    parser.add_argument("-z", "--zindexes", type=int, action='store', dest="zindexes", default=-1)
    parser.add_argument("-t", "--resthresh", type=float, dest="max_thresh", default=1.5, action='store', help="maximum residual to allow")
    args = parser.parse_args()
    print(args)

def keep_which(stk, peaks, w=3, sz = 7):
    """
    Depricated
    Only necessary if grabbing substks to prevent accessing out of bounds pixels.
    """
    imsz = stk.shape
    keep_list = []
    for ii, p in enumerate(peaks):
        if (p[0] <= w) or (p[0] >= imsz[0]-w):
            continue
        if (p[1] <= w) or (p[1] >= imsz[1]-w):
            continue
        if (p[2] <= w) or (p[2] >= imsz[2]-w):
            continue
        keep_list.append(ii)
    return keep_list

def find_beads_3D(fnames_list, bead_template, match_threshold=0.65, upsamp_factor = 5):
    """
    3D registration from sparse bead images.
    Parameters
    ----------
    fnames_dict : dict
        Dictionary (hybe name:list filenames) to load bead images
    bead_template : numpy.array
        Array of normalized values of 'average bead' intensities.
        This helps to not pick up noisy spots/hot pixels.
    ref_reference : str - default('hybe1')
        Which hybe is the reference destination to map source hybes onto.
    max_dist : int - default(50)
        The maximum distanced allowed between beads found when pairing 
        beads between multiple hybes.
    match_threshold : float - default(0.75)
        The similarity threshold between bead_template and images to 
        consider potential beads.
    Returns
    -------
    tforms : dict
        Dictionary of translation vectors (x, y, z)? maybe (y, x, z)
    """

    ref_stk = np.stack([io.imread(i) for i in fnames_list], axis=2).astype('float')
    ref_match = match_template(ref_stk, bead_template, pad_input=True)
    ref_beads = peak_local_max(ref_match, threshold_abs=match_threshold)
    upsamp_bead = resize(bead_template[2:5, 2:5, 1:4],
                         (3*upsamp_factor, 3*upsamp_factor, 3*upsamp_factor),
                         mode='constant',anti_aliasing=False)
    subpixel_beads = []
    for y, x, z in ref_beads:
        substk = ref_stk[y-5:y+6, x-5:x+6, z-2:z+3]
        if substk.shape[0] != 11 or substk.shape[1] != 11:
            continue # candidate too close to edge
        try:
            upsamp_substk = resize(substk,
                                   (substk.shape[0]*upsamp_factor,
                                    substk.shape[1]*upsamp_factor,
                                    substk.shape[2]*upsamp_factor),
                                   mode='constant',anti_aliasing=False)
        except:
            continue
        bead_match = match_template(upsamp_substk,
                                    upsamp_bead, pad_input=True)
        yu, xu, zu = np.where(bead_match==bead_match.max())
        yu = (yu[0]-int(upsamp_substk.shape[0]/2))/upsamp_factor
        xu = (xu[0]-int(upsamp_substk.shape[1]/2))/upsamp_factor
        zu = (zu[0]-int(upsamp_substk.shape[2]/2))/upsamp_factor
        ys, xs, zs = (yu+y, xu+x, zu+z)
        subpixel_beads.append((ys, xs, zs))

    return subpixel_beads

def ensembl_bead_reg(bead_dict,pos, reg_ref='hybe1', max_dist=200,
                     dbscan_eps=3, dbscan_min_samples=20):
    """
    Calculate transformation to register 2 sets of images together
    Given a set of candidate bead coordinates (xyz) and a reference hybe find min-error translation.
    
    Parameters
    ----------
    bead_dict : dict
    pos: str
    
    Returns
    -------
    tform_dict : dict
      key - hybe name
      value - tuple(translation, residual, number beads)
    
    This task is trivial given a paired set of coordinates for source/destination beads. 
    The input to this function is unpaired so the first task is pairing beads. Given the 
    density of beads in relation to the average distance between beads it is not reliable to 
    simply use closest-bead-candidate pairing. However, for each bead in the destination we can find 
    all beads within from distance in the source set of beads and calculate the difference of all these pairs.
    Bead pairs that are incorrect have randomly distributed differences however correct bead pairs all 
    have very similar differences. So a density clustering of the differences is performed to identify 
    the set of bead pairings representing true bead pairs between source/destination.
    
    The best translation is found my minimizing the mean-squared error between source/destination after pairing.
    """
    # Final optimization objective function
    def error_func(translation):
        fit = np.add(translation, dest)
        fit_error = np.sqrt(np.subtract(ref, fit)**2)
        fit_error = np.mean(fit_error)
        return fit_error
    
    # Perform pairing by density clustering differences between all possible pairs
    bead_dict = bead_dict.copy()
    hybes = list(bead_dict.keys())
    ref = [i for i in hybes if reg_ref in i][0]
    ref_beadarray = bead_dict.pop(ref)
    if len(ref_beadarray)<dbscan_min_samples:
        return 'Failed position not enough reference beads found.'
    ref_beadarray = np.stack(ref_beadarray, axis=0)
    if ref_beadarray.shape[0]<dbscan_min_samples:
        return 'Failed position not enough reference beads found.'
    ref_tree = KDTree(ref_beadarray[:, :2])
    tform_dict = {ref: (np.array((0, 0, 0)), 0, float('inf'))}
    db_clusts = DBSCAN(min_samples=dbscan_min_samples, eps=dbscan_eps)
    for h, beadarray in bead_dict.items():
        if len(beadarray)<dbscan_min_samples:
            tform_dict[h] = 'Not enough bead pairs found.'
            print(len(beadarray))
            continue
        beadarray = np.stack(beadarray, axis=0)
        t_est = []
        ref_beads = []
        dest_beads = []
        close_beads = ref_tree.query_ball_point(beadarray[:, :2], r=max_dist)
        for i, bead in zip(close_beads, beadarray):
            if len(i)==0:
                continue
            for nbor in i:
                t = ref_beadarray[nbor]-bead
                t_est.append(np.subtract(ref_beadarray[nbor], bead))
                ref_beads.append(ref_beadarray[nbor])
                dest_beads.append(bead)
        ref_beads = np.array(ref_beads)
        dest_beads = np.array(dest_beads)
        if len(t_est)<dbscan_min_samples:
            tform_dict[h] = 'Not enough bead pairs found.'
            continue
        t_est = np.stack(t_est, axis=0)
        db_clusts.fit(t_est)
        most_frequent_cluster = Counter(db_clusts.labels_)
        most_frequent_cluster.pop(-1)
        try:
            most_frequent_cluster = most_frequent_cluster.most_common(1)[0][0]
        except IndexError:
            tform_dict[h] = 'Not enough bead pairs found.'
            continue
        paired_beads_idx = db_clusts.labels_==most_frequent_cluster
        ref = ref_beads[paired_beads_idx]
        dest = dest_beads[paired_beads_idx]
        t_est = t_est[paired_beads_idx]
        tform_dict[h]=list(zip(ref_beads, dest_beads))
        
        # Optimize translation to map paired beads onto each other
        opt_t = scipy.optimize.fmin(error_func, np.mean(t_est, axis=0), full_output=True, disp=False)
        tform_dict[h] = (opt_t[0], opt_t[1], sum(paired_beads_idx))
    return tform_dict

def classify_tform(tform_dict,pos,max_thresh):
    """
    Determine if a position passes or fails
    
    Parameters
    ----------
    tform_dict: dict
        found tforms
    pos: str
        position name
    max_thresh:float
        threshold for max average pixle displacement to pass
       
    Returns
    -------
    goodness: int
        0 if pass
        >0 if fail
    """
    goodness = 0
    if type(tform_dict) != dict:
        print('No beads in', pos)
        goodness = 1
    else:
        for hybe in tform_dict.keys():
            print(tform_dict[hybe][1])
            if hybe != 'nucstain':
                if isinstance(tform_dict[hybe][0], str):
                    goodness = goodness + 1
                    print(pos, hybe, 'not enough bead pairs found')
                elif tform_dict[hybe][1] > max_thresh:
                    goodness = goodness + 1
                    print(pos, hybe, 'residual is too high',tform_dict[hybe][1])
    return goodness

def add_bead_data(bead_dicts, ave_bead, Input):
    """
    Main Bead and Tform Wrapper
    
    Parameters
    ----------
    bead_dicts: dict 
        previously found beads
    ave_bead: ndarray
        template of top half of bead
    Input: dict
        contains the file names of the bead images
        contains the name of the position
        
    Returns
    -------
    bead_dicts: dictionary 
        found beads
    tform_dicts: dictionary
        found transformations
    pos: str
        position name
    processed: int
        0 if nothing new found
        >0 if new beads or tforms are found
    """
    fnames_dict = Input['fname_dicts']
    pos = Input['posnames']
    print('Starting ', pos)
    if pos in bead_dicts:
        bead_dict = bead_dicts[pos]
    else:
        bead_dict = {}
    processed = 0   
    for h in fnames_dict.keys():
        #convert to hybe1 not hybe1_4
        H = h.split('_')[0]
        if H in bead_dict:
            continue
#         if 'nucstain' in H:
#             continue
        else:
            processed = processed + 1
            beads = find_beads_3D(fnames_dict[h], ave_bead)
            bead_dict[H] = beads
    if processed == 0:
        tform_dict = []
    else:
        tform_dict = ensembl_bead_reg(bead_dict,pos)
    
    return bead_dict, tform_dict, pos, processed
    
def load_fnames(md_path,zindexes):
    md = Metadata(md_path)
    md.image_table = md.image_table[[True if (('hybe' in i) or ('nucstain' in i)) else False for i in md.image_table.acq]]
    posnames = md.posnames
    # Speed up checking if a position is already fully done
    not_finished = []
    finished_pos = []
    for pos in posnames:
        try:
            processing = pickle.load(open(os.path.join(md_path,'codestacks',pos,'processing.pkl'),'rb'))
            if len(processing)==18: # Check to make sure position is finished
                finished_pos.append(pos)
            else:
                not_finished.append(pos)
        except:
            not_finished.append(pos)
    random_posnames = np.random.choice(not_finished, size=len(not_finished), replace=False)
    print('posnames loaded')
    hybe_list = sorted([i.split('_')[0] for i in md.acqnames if ('hybe' in i) or ('nucstain' in i)])
    if zindexes == -1:
        Input = [ {'fname_dicts': md.stkread(Channel='DeepBlue', Position=pos,
                           fnames_only=True, groupby='acq', 
                          hybe=hybe_list), 'posnames': pos} for pos in random_posnames]
    else:
        Input = [ {'fname_dicts': md.stkread(Channel='DeepBlue', Position=pos,
                           fnames_only=True, groupby='acq', 
                          hybe=hybe_list, Zindex=['range',1,zindexes]), 'posnames': pos} for pos in random_posnames]
#         Input = []
#         for pos in random_posnames:
#             pos_df = {}
#             for hybe in hybe_list:
#                 pos_df[hybe] = []
#             for z in range(1,zindexes):
#                 fnames = md.stkread(Channel='DeepBlue', Position=pos,
#                                     fnames_only=True, groupby='acq', 
#                                     hybe=hybe_list,Zindex=z)
#                 for h,f in fnames.items():
#                     pos_df[h.split('_')[0]].append(f)
#             Input.append({'fname_dicts':pos_df,'posnames': pos})
    print('fnames loaded')
    return Input
                
if __name__ == '__main__':
    #Setting up paths
    analysis_path = args.analysis_path
    md_path = args.md_path
    zindexes = args.zindexes
    ncpu = args.ncpu
    max_thresh = args.max_thresh
    print(args)
    
    #Set up ouput location
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    results_path = os.path.join(analysis_path,'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #Create Bead Template
    bead = np.zeros((7, 7, 5))
    bead[3, 3, 2] = 1
    bead = gaussian(bead, (1.5, 1.5, 0.85))
    Ave_Bead = bead/bead.max()
    
    #Setting up parrallel pool
    os.environ['MKL_NUM_THREADS'] = '3'
    os.environ['GOTO_NUM_THREADS'] = '3'
    os.environ['OMP_NUM_THREADS'] = '3'
    
    #Finding and loading beads and tforms
    if os.path.exists(os.path.join(results_path,'beads.pkl')):
        bead_dicts = pickle.load(open(os.path.join(results_path,'beads.pkl'), 'rb'))
        print('bead_dict Loaded')
    else:
        bead_dicts = defaultdict(dict)
    if os.path.exists(os.path.join(results_path,'tforms.pkl')):
        tform_dicts = pickle.load(open(os.path.join(results_path,'tforms.pkl'), 'rb'))
        print('tform_dict Loaded')
    else:
        tform_dicts = defaultdict(dict)
        
    #Retreiving file names and position names
    Input = load_fnames(md_path,zindexes)
    
    #Find Beads and Tforms
    pfunc = partial(add_bead_data,bead_dicts,Ave_Bead)
    with multiprocessing.Pool(ncpu) as p:
        for Bead_dict,Tform_dict,pos,processed in p.imap(pfunc, Input, chunksize=1):
            if processed>0:
                bead_dicts[pos] = Bead_dict
                pickle.dump(bead_dicts, open(os.path.join(results_path,'beads.pkl'), 'wb'))
                goodness = classify_tform(Tform_dict,pos,max_thresh)
                if goodness == 0:
                    tform_dicts['good'][pos] = Tform_dict
                    print(pos, 'all good')
                    if pos in tform_dicts['bad'].keys():
                        del tform_dicts['bad'][pos]
                else:
                    if pos in tform_dicts['good'].keys():
                        del tform_dicts['good'][pos]
                    tform_dicts['bad'][pos] = Tform_dict
                    print(pos, 'no bueno')
                pickle.dump(tform_dicts, open(os.path.join(results_path,'tforms.pkl'), 'wb'))
            else:
                print(pos,' Already Done')