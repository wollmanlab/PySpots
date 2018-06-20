from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from collections import Counter, defaultdict
import numpy as np
import pickle
import scipy
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bead_path", type=str, help="Path pickle dictionary of candidate beads per position name.")
    parser.add_argument("out_path", type=str, help="Path to save output.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=4, action='store', nargs=1, help="Number of cores to utilize (default 4).")
    parser.add_argument("-t", "--resthresh", type=float, dest="max_thresh", default=1.5, action='store', nargs=1, help="maximum residual to allow")
    args = parser.parse_args()
    print(args)

def ensembl_bead_reg(hybe_dict, reg_ref='hybe1', max_dist=200,
                     dbscan_eps=3, dbscan_min_samples=20):
    """
    Given a set of candidate bead coordinates (xyz) and a reference hybe find min-error translation.
    
    Parameters
    ----------
    hybe_dict : dict
    
    Returns
    -------
    results : dict
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
    hybe_dict = hybe_dict.copy()
    hybes = list(hybe_dict.keys())
    ref = [i for i in hybes if reg_ref in i][0]
    ref_beadarray = hybe_dict.pop(ref)
    if ref_beadarray.shape[0]<dbscan_min_samples:
        return 'Failed position not enough reference beads found.'
    ref_tree = KDTree(ref_beadarray[:, :2])
    results = {ref: (np.array((0, 0, 0)), 0, float('inf'))}
    db_clusts = DBSCAN(min_samples=dbscan_min_samples, eps=dbscan_eps)
    for h, beadarray in hybe_dict.items():
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
            results[h] = 'Not enough bead pairs found.'
            continue
        t_est = np.stack(t_est, axis=0)
        db_clusts.fit(t_est)
        most_frequent_cluster = Counter(db_clusts.labels_)
        most_frequent_cluster.pop(-1)
        try:
            most_frequent_cluster = most_frequent_cluster.most_common(1)[0][0]
        except IndexError:
            results[h] = 'Not enough bead pairs found.'
            continue
        paired_beads_idx = db_clusts.labels_==most_frequent_cluster
        ref = ref_beads[paired_beads_idx]
        dest = dest_beads[paired_beads_idx]
        t_est = t_est[paired_beads_idx]
        results[h]=list(zip(ref_beads, dest_beads))
        # Optimize translation to map paired beads onto each other
        opt_t = scipy.optimize.fmin(error_func, np.mean(t_est, axis=0), full_output=True, disp=False)
        results[h] = (opt_t[0], opt_t[1], sum(paired_beads_idx))
    return results

if __name__ == '__main__':
    hybe_dict = pickle.load(open(args.bead_path,'rb'))
    tforms_dict = defaultdict(dict)
    tforms_dict['good'] = defaultdict(dict)
    tforms_dict['bad'] = defaultdict(dict)
    for pos in hybe_dict.keys():
        results = ensembl_bead_reg(hybe_dict[pos])
        goodness = 0
        if type(results) != dict:
            print('No beads in', pos)
            continue
        else:
            for hybe in results.keys():
                if type(results[hybe][0]) == str:
                    goodness = goodness + 1
                    print(pos, hybe, 'not enough bead pairs found')
                elif results[hybe][1] > args.max_thresh:
                    goodness = goodness + 1
                    print(pos, hybe, 'residual is too high',results[hybe][1]) 
            if goodness == 0:
                tforms_dict['good'][pos] = results
                print(pos, 'all good')
            else:
                tforms_dict['bad'][pos] = results
                print(pos, 'no bueno')
    print('Finished finding tforms')
    pickle.dump(tforms_dict,open(os.path.join(args.out_path,'tforms.pkl'),'wb'))