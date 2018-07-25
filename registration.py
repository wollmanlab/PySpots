import numpy
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from metadata import Metadata
from skimage import io
from skimage.feature import match_template, peak_local_max

def find_beads_3D(fnames_list, bead_template, match_threshold=0.75):
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
    hybe_names = ['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6']
    #fnames_dict = {k.split('_')[0]:v for k, v in fnames_dict.items()}
#     fish_md = Metadata(md_path, md_name='Metadata.dsv')
#     pos = fish_md.posnames[1]
#     fish_md.image_table = fish_md.image_table[fish_md.image_table.Position==pos]
    #tforms = {reg_reference: {'tvec': numpy.array([0, 0])}}
    #ref_fnames = fnames_dict.pop(reg_reference)
    ref_stk = numpy.stack([io.imread(i) for i in fnames_list], axis=2).astype('float')
    ref_match = match_template(ref_stk, bead_template)
    ref_beads = peak_local_max(ref_match, threshold_abs=match_threshold)
    ref_substks, keep_list = fetch_substacks(ref_stk, ref_beads, w=3)
    ref_beads = ref_beads[keep_list, :]
    #del ref_stk
    return ref_beads, ref_substks

def processStk(stk, npeaks=200):
    stk = np.subtract(stk, np.min(stk, axis=2)[:,:,np.newaxis])
    stk = gaussian_filter(stk, (0.9, 0.9, 0.9))
    peaks = peak_local_max(stk, num_peaks=npeaks)
    return stk, peaks

def fetch_substacks(stk, peaks, w=3, sz = 7):
    imsz = stk.shape
    substks = {}
    keep_list = []
    for ii, p in enumerate(peaks):
        if (p[0] <= w) or (p[0] >= imsz[0]-w):
            #pop_list.append(ii)
            continue
        if (p[1] <= w) or (p[1] >= imsz[1]-w):
            #pop_list.append(ii)
            continue
        if (p[2] <= w) or (p[2] >= imsz[2]-w):
            #pop_list.append(ii)
            continue
        substk = stk[p[0]:p[0]+sz, p[1]:p[1]+sz, p[2]:p[2]+sz]
        substks[tuple(p)] = substk
        keep_list.append(ii)
    #pop_list = {tuple(k):0 for k in pop_list}
    return substks, keep_list

def find_average_bead_profile(stk, w, sz):
    fstk, peaks = processStk(stk.copy())
    s = fetch_substacks(stk, peaks)
    beads = []
    for p in peaks:
        p = tuple(p)
        if p[2] == 18:
            beads.append(s[p])
    ave_bead = np.mean(beads, 0)
    ave_bead = ave_bead/np.amax(ave_bead)

def reject_outliers(data, m=3):
    d = np.abs(data - np.median(data, axis=0))
    mdev = np.median(d, axis=0)
    s = [True if sum(i)<m else False for i in d]
    data = [d for idx, d in enumerate(data) if s[idx]]
    return data

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
    return nbeads, tforms_xy, tforms_z, error_xy, error_z, master_beads

from codestack_creation import tform_image
def pseudo_bead_stacks_byposition(posname, md_pth, tforms_xy, tforms_z, zstart=6, k=2, reg_ref='hybe1'):
    hybe_names = ['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6']
    md = Metadata(md_pth)
    xy = tforms_xy[posname]
    z = tforms_z[posname]
    z = {k: int(np.round(np.mean(v))) for k, v in z.items()}
    z[reg_ref] = 0
    xy[reg_ref] = (0,0)
    cstk = []
    chan='DeepBlue'
    for hybe in hybe_names:
        t = xy[hybe]
        zindexes = list(range(zstart-z[hybe]-k, zstart-z[hybe]+k+1))
        print(zindexes)
        zstk = md.stkread(Channel=chan, hybe=hybe, Position=posname, Zindex=zindexes)
        zstk = zstk.max(axis=2)
        zstk = tform_image(zstk, chan, t)
        cstk.append(zstk)
        del zstk
    cstk = np.stack(cstk, axis=2)
    #nf = np.percentile(cstk, 90, axis=(0, 1))
    return cstk
