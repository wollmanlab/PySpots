import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree

def find_tforms(beads, reg_ref_hybe = 'hybe1', max_dist = 30, verbose=True):
    """
    Process putative bead coordinates to find registration shifts between hybes.
    
    Parameters
    ----------
    beads : dict
        Dictionary of bead coordinates for each hybe
    reg_ref_hybe : str - default - 'hybe1'
    max_dist : int
        Maximum distance for two beads to be linked between hybes
    
    Returns
    -------
    Not well defined yet
    
    Notes - not very clean function - future work cleans it up and probably 
    implements a better way of finding Z fits.
    
    """
    hybe_names = []
    tforms = {reg_ref_hybe: {'tvec': numpy.array([0, 0])}}
    ref_beads = beads.pop(reg_ref_hybe)
    bead_sets = defaultdict(dict)
    # Find beads in nonreference hybes
    tree = KDTree(ref_beads[:, 0:2]) #Data structure for finding close beads quickly
    for h, current_beads in beads.items():
        hybe_names.append(h)
        if verbose:
            print('Registering:', h)
        # Reject bead candidates that are >max_dist from any bead in reference
        dists, idxes = tree.query(current_beads[:, 0:2])
        shifts = []
        # Store all bead candidates in nonreference with their closest
        # cognate bead in the reference
        for b_i, b in enumerate(current_beads):
            if dists[b_i]<max_dist:
                bead_sets[tuple(ref_beads[idxes[b_i]])][h] = b
    # Only keep beads found in all hybes 
    master_beads = {}
    for k, v in bead_sets.items():
        if len(v)==len(hybe_names):
            master_beads[k] = v  
    tforms_xy = defaultdict(list)
    tforms_z = defaultdict(list)
    tforms_z_raw = defaultdict(list)
    # Iterate over all beads found in all hybes and fit gaussians then
    # find the shift between the reference and nonreference
    for k, hybe_dict in master_beads.items():
        #if str(k) in ref_substks:
            #sstk = ref_substks[str(k)]
            #ref_fit = gaussfitter.gaussfit(sstk[:,:,3])
        ref_fit = (k[0], k[1])
            #print(ref_fit)
        for h, dest_bead in hybe_dict.items():
                #if str(tuple(dest_bead)) in destination_substks[h]:
            try:
                    #sstk = destination_substks[h][str(tuple(dest_bead))]
                    #dest_fit = gaussfitter.gaussfit(sstk[:,:,3])
                    #dest_fit = (dest_bead[0]+dest_fit[2]-3, dest_bead[1]+dest_fit[3]-3)
                dest_fit = (dest_bead[0], dest_bead[1])
                tforms_xy[h].append(np.subtract(ref_fit, dest_fit))
                tforms_z[h].append(np.subtract(k[2], dest_bead[2]))
                tforms_z_raw[h].append((k[2], dest_bead[2]))
            except:
                continue


    # Attempt to find robust mean of transformation to map beads between
    # source/destination if fails use regular mean
    tforms_z_Ave = defaultdict(list)
    try:
        tforms_z_r = {h: huber(v).astype('int') for h, v in tforms_z.items()}
    except:
        tforms_z_r = {h: np.round(np.mean(v)).astype('int') for h, v in tforms_z.items()}
    #tforms_z_Ave = {h: np.mean(reject_outliers(v)) for h, v in tforms_z.items()}
    tforms_xy = {h: np.mean(reject_outliers(v), axis=0) for h, v in tforms_xy.items()}
    error_xy = 0
    error_z = 0
    nbeads = len(master_beads)
    return nbeads, tforms_xy, tforms_z, error_xy, error_z
